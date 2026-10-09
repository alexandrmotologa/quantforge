"""Pipeline runner executing end-to-end quantization recipes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import yaml

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import (
    ImatrixConfig,
    MatrixQuantConfig,
    PerplexityConfig,
    QuantizationConfig,
    QuantType,
)
from quantforge.engines.imatrix import ImatrixEngine, ImatrixResult
from quantforge.engines.pareto import ParetoAnalyzer, QuantPoint, ParetoReport
from quantforge.engines.perplexity import PerplexityEngine, PerplexityResult
from quantforge.engines.quantize import QuantizeEngine, QuantizationResult
from quantforge.formats.card_generator import ModelCardGenerator
from quantforge.formats.gguf_reader import GGUFReader, GGUFModelInfo


@dataclass
class PipelineProgress:
    """Status updates across pipeline stages."""

    stage: str
    current_step: int
    total_steps: int
    detail: str


@dataclass
class PipelineExecutionSummary:
    """Consolidated results of an automated pipeline run."""

    recipe_name: str
    input_model: Path
    output_dir: Path
    success: bool
    imatrix_result: Optional[ImatrixResult] = None
    quant_results: List[QuantizationResult] = field(default_factory=list)
    eval_results: List[PerplexityResult] = field(default_factory=list)
    pareto_report: Optional[ParetoReport] = None
    model_card_path: Optional[Path] = None
    error_message: Optional[str] = None


class PipelineRunner:
    """Executes multi-step automated quantization recipes."""

    def __init__(self, binary_manager: Optional[BinaryManager] = None) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.quant_engine = QuantizeEngine(self.binary_manager)
        self.imatrix_engine = ImatrixEngine(self.binary_manager)
        self.ppl_engine = PerplexityEngine(self.binary_manager)
        self.pareto_analyzer = ParetoAnalyzer()
        self.card_gen = ModelCardGenerator()

    async def run_recipe_file(
        self,
        recipe_path: Path,
        override_output_dir: Optional[Path] = None,
        on_progress: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> PipelineExecutionSummary:
        """Loads and executes a recipe from a YAML file."""
        recipe_file = Path(recipe_path).resolve()
        if not recipe_file.is_file():
            raise FileNotFoundError(f"Recipe file not found: {recipe_file}")

        with open(recipe_file, "r", encoding="utf-8") as f:
            recipe_dict = yaml.safe_load(f)

        return await self.run_recipe(recipe_dict, override_output_dir, on_progress)

    async def run_recipe(
        self,
        recipe: Dict[str, Any],
        override_output_dir: Optional[Path] = None,
        on_progress: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> PipelineExecutionSummary:
        """Executes the pipeline stages defined in the recipe dictionary."""
        name = recipe.get("name", "quantforge-pipeline")
        input_model = Path(recipe["input_model"]).resolve()
        output_dir = (override_output_dir or Path(recipe.get("output_dir", "./dist"))).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_model.is_file():
            raise FileNotFoundError(f"Base input model not found: {input_model}")

        # Step 0: Inspect input model info
        model_info: Optional[GGUFModelInfo] = None
        try:
            reader = GGUFReader(input_model)
            model_info = reader.read_model_info(load_tensors=False)
        except Exception:
            pass

        summary = PipelineExecutionSummary(
            recipe_name=name,
            input_model=input_model,
            output_dir=output_dir,
            success=False,
        )

        total_steps = 4
        current_step = 1

        def _notify(stage: str, detail: str) -> None:
            if on_progress:
                on_progress(
                    PipelineProgress(
                        stage=stage,
                        current_step=current_step,
                        total_steps=total_steps,
                        detail=detail,
                    )
                )

        # Stage 1: Imatrix Calibration (if requested)
        calib_cfg = recipe.get("calibration")
        imatrix_path: Optional[Path] = None
        if calib_cfg:
            _notify("calibration", "Calibrating importance matrix...")
            data_file = calib_cfg.get("dataset")
            if data_file:
                calib_data_path = Path(data_file).resolve()
            else:
                # Generate sample corpus if none provided
                calib_data_path = output_dir / "calibration_corpus.txt"
                self.imatrix_engine.generate_sample_calibration_corpus(calib_data_path)

            out_imatrix = output_dir / f"{input_model.stem}.dat"
            im_config = ImatrixConfig(
                input_path=input_model,
                data_path=calib_data_path,
                output_path=out_imatrix,
                ctx_size=calib_cfg.get("ctx_size", 2048),
                chunks=calib_cfg.get("chunks", 64),
                n_gpu_layers=calib_cfg.get("n_gpu_layers", 0),
            )
            imatrix_res = await self.imatrix_engine.calibrate(im_config)
            summary.imatrix_result = imatrix_res
            if imatrix_res.success:
                imatrix_path = out_imatrix
            else:
                summary.error_message = f"Calibration failed: {imatrix_res.error_message}"
                return summary

        current_step += 1

        # Stage 2: Quantization Matrix
        _notify("quantize", "Executing quantizations...")
        raw_quants = recipe.get("quants", ["Q4_K_M", "Q5_K_M", "Q8_0"])
        quant_enums: List[QuantType] = []
        for q in raw_quants:
            try:
                quant_enums.append(QuantType(q))
            except ValueError:
                pass

        matrix_cfg = MatrixQuantConfig(
            input_path=input_model,
            output_dir=output_dir,
            quant_types=quant_enums,
            imatrix_path=imatrix_path,
        )
        quant_results = await self.quant_engine.quantize_matrix(matrix_cfg)
        summary.quant_results = quant_results

        current_step += 1

        # Stage 3: Evaluation (if configured)
        eval_cfg = recipe.get("evaluation")
        points: List[QuantPoint] = []

        if eval_cfg and eval_cfg.get("dataset"):
            _notify("evaluation", "Evaluating perplexity...")
            eval_dataset = Path(eval_cfg["dataset"]).resolve()

            # Baseline eval on input model if unquantized
            base_ppl = None
            if eval_cfg.get("eval_baseline", True):
                base_eval = await self.ppl_engine.evaluate(
                    PerplexityConfig(
                        model_path=input_model,
                        data_path=eval_dataset,
                        ctx_size=eval_cfg.get("ctx_size", 2048),
                    )
                )
                if base_eval.success and base_eval.perplexity:
                    base_ppl = base_eval.perplexity
                    points.append(
                        QuantPoint(
                            quant_type=model_info.dominant_quant or "BASE",
                            file_path=input_model,
                            file_size_gb=input_model.stat().st_size / (1024**3),
                            perplexity=base_ppl,
                            is_baseline=True,
                        )
                    )

            # Evaluate each successful quant
            for qr in quant_results:
                if not qr.success or not qr.output_path.is_file():
                    continue
                p_eval = await self.ppl_engine.evaluate(
                    PerplexityConfig(
                        model_path=qr.output_path,
                        data_path=eval_dataset,
                        ctx_size=eval_cfg.get("ctx_size", 2048),
                    ),
                    baseline_ppl=base_ppl,
                )
                summary.eval_results.append(p_eval)
                if p_eval.success and p_eval.perplexity:
                    points.append(
                        QuantPoint(
                            quant_type=qr.quant_type.value,
                            file_path=qr.output_path,
                            file_size_gb=qr.output_path.stat().st_size / (1024**3),
                            perplexity=p_eval.perplexity,
                        )
                    )

            if points and eval_cfg.get("calc_pareto", True):
                summary.pareto_report = self.pareto_analyzer.compute_frontier(points)

        current_step += 1

        # Stage 4: Model Card Generation
        if recipe.get("export_model_card", True):
            _notify("export", "Exporting Model Card...")
            card_md = self.card_gen.generate_card(
                model_name=input_model.stem,
                model_info=model_info,
                pareto_report=summary.pareto_report,
                quant_points=points if points else [
                    QuantPoint(
                        quant_type=qr.quant_type.value,
                        file_path=qr.output_path,
                        file_size_gb=qr.quantized_size_bytes / (1024**3),
                        perplexity=0.0,
                    )
                    for qr in quant_results if qr.success
                ],
            )
            card_path = output_dir / "README.md"
            card_path.write_text(card_md, encoding="utf-8")
            summary.model_card_path = card_path

        summary.success = any(qr.success for qr in quant_results)
        return summary
