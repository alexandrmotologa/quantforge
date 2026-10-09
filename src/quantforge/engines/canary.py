"""Canary and functional quality degeneration benchmark for quantized GGUF models."""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.executor import SubprocessExecutor

logger = logging.getLogger(__name__)


@dataclass
class CanaryTest:
    """Descriptor for a single validation prompt and evaluation criterion."""

    name: str
    category: str
    prompt: str
    max_tokens: int = 256
    validator: Optional[Callable[[str], bool]] = None


@dataclass
class CanaryTestResult:
    """Outcome of an individual canary test."""

    name: str
    category: str
    passed: bool
    output_text: str
    score: float = 1.0
    latency_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class CanaryReport:
    """Aggregated evaluation report across all canary test cases."""

    model_path: Path
    total_tests: int
    passed_tests: int
    pass_rate_pct: float
    repetition_ratio: float
    duration_seconds: float
    test_results: List[CanaryTestResult] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable overview of the canary benchmark results."""
        lines = [
            f"Canary Quality Benchmark for {self.model_path.name}:",
            f"  • Pass Rate:       {self.pass_rate_pct:.1f}% ({self.passed_tests}/{self.total_tests} passed)",
            f"  • Diversity Ratio: {self.repetition_ratio:.2f} (Unique 4-grams)",
            f"  • Total Time:      {self.duration_seconds:.2f}s",
            "--------------------------------------------------",
        ]
        for t in self.test_results:
            status = "✓ PASS" if t.passed else "✗ FAIL"
            lines.append(f"  [{status}] {t.name:<20} (Score: {t.score:.2f})")
            if not t.passed and t.error_message:
                lines.append(f"         Reason: {t.error_message}")
        return "\n".join(lines)


# Validation helper functions
def _validate_json(output: str) -> bool:
    """Attempts to extract and parse JSON payload."""
    match = re.search(r"(\{.*\})", output, re.DOTALL)
    candidate = match.group(1) if match else output.strip()
    try:
        data = json.loads(candidate)
        return isinstance(data, dict) and "status" in data
    except Exception:
        return False


def _validate_python_code(output: str) -> bool:
    """Extracts Python code blocks or raw script and tests syntax validity."""
    match = re.search(r"```python(.*?)```", output, re.DOTALL)
    code = match.group(1) if match else output
    try:
        ast.parse(code.strip())
        return "fibonacci" in code
    except Exception:
        return False


def _calculate_4gram_diversity(text: str) -> float:
    """Calculates ratio of unique 4-grams to detect repetition degeneration."""
    words = [w.lower() for w in re.findall(r"\w+", text)]
    if len(words) < 4:
        return 1.0
    fourgrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    if not fourgrams:
        return 1.0
    return len(set(fourgrams)) / len(fourgrams)


def _validate_math_logic(output: str) -> bool:
    """Checks for the correct answer '8' to the farmer sheep riddle."""
    text_lower = output.lower()
    return bool(re.search(r"\b8\b", text_lower)) and "7" not in text_lower[-40:]


# Standard battery of zero-shot canary tests
STANDARD_CANARY_TESTS: List[CanaryTest] = [
    CanaryTest(
        name="structured_json",
        category="Formatting & Schema",
        prompt="Generate a valid JSON object with keys 'status': 'ok', 'code': 200, 'tags': ['ai', 'quant']. Output JSON only, no markdown.",
        max_tokens=128,
        validator=_validate_json,
    ),
    CanaryTest(
        name="code_syntax",
        category="Code Generation",
        prompt="Write a short Python function named `fibonacci(n)` that returns the nth fibonacci number. Output only valid Python code in a codeblock.",
        max_tokens=256,
        validator=_validate_python_code,
    ),
    CanaryTest(
        name="math_reasoning",
        category="Reasoning & Logic",
        prompt="A farmer has 15 sheep, and all but 8 die. How many sheep are left alive? Think briefly and state the final number clearly.",
        max_tokens=128,
        validator=_validate_math_logic,
    ),
    CanaryTest(
        name="repetition_check",
        category="Coherence & Fluency",
        prompt="Explain how photosynthesis works in green plants in two clear and informative paragraphs.",
        max_tokens=300,
        validator=lambda text: _calculate_4gram_diversity(text) >= 0.75,
    ),
]


class CanaryEngine:
    """Executes zero-shot canary benchmarks against a quantized GGUF model via llama-cli."""

    def __init__(
        self,
        binary_manager: Optional[BinaryManager] = None,
        executor: Optional[SubprocessExecutor] = None,
    ) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.executor = executor or SubprocessExecutor()

    def build_command(
        self,
        model_path: Union[str, Path],
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.1,
        threads: Optional[int] = None,
    ) -> List[str]:
        """Constructs CLI arguments for llama-cli inference execution."""
        binary_info = self.binary_manager.find_binary("cli")
        if not binary_info.found or not binary_info.path:
            raise FileNotFoundError(
                "Executable 'llama-cli' was not found in your environment or llama.cpp path."
            )

        cmd = [
            str(binary_info.path),
            "-m",
            str(Path(model_path).resolve()),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "-c",
            "1024",
            "--log-disable",
        ]

        if threads and threads > 0:
            cmd.extend(["-t", str(threads)])

        return cmd

    async def run_benchmark(
        self,
        model_path: Union[str, Path],
        tests: Optional[List[CanaryTest]] = None,
        threads: Optional[int] = None,
        temperature: float = 0.1,
    ) -> CanaryReport:
        """Executes test battery and aggregates pass rates and repetition scores."""
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        test_battery = tests or STANDARD_CANARY_TESTS
        results: List[CanaryTestResult] = []
        overall_start = time.time()
        diversity_scores: List[float] = []

        for test in test_battery:
            cmd = self.build_command(
                model_path=path,
                prompt=test.prompt,
                max_tokens=test.max_tokens,
                temperature=temperature,
                threads=threads,
            )

            t_start = time.time()
            proc = await self.executor.run(cmd)
            t_duration = time.time() - t_start

            raw_output = proc.stdout.strip()
            # Clean prompt echo if llama-cli repeats it
            if raw_output.startswith(test.prompt):
                raw_output = raw_output[len(test.prompt) :].strip()

            passed = False
            error_msg = None

            if proc.return_code != 0:
                error_msg = f"llama-cli exited with code {proc.return_code}: {proc.stderr[:100]}"
            elif test.validator:
                try:
                    passed = bool(test.validator(raw_output))
                    if not passed:
                        error_msg = "Validator check failed on model output."
                except Exception as exc:
                    error_msg = f"Validator raised error: {exc}"
            else:
                passed = len(raw_output) > 0

            # Measure diversity score for long texts
            div = _calculate_4gram_diversity(raw_output)
            diversity_scores.append(div)

            score = 1.0 if passed else 0.0

            results.append(
                CanaryTestResult(
                    name=test.name,
                    category=test.category,
                    passed=passed,
                    output_text=raw_output,
                    score=score,
                    latency_seconds=t_duration,
                    error_message=error_msg,
                )
            )

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = round((passed_count / max(total, 1)) * 100.0, 1)
        avg_diversity = sum(diversity_scores) / max(len(diversity_scores), 1)

        return CanaryReport(
            model_path=path,
            total_tests=total,
            passed_tests=passed_count,
            pass_rate_pct=pass_rate,
            repetition_ratio=round(avg_diversity, 2),
            duration_seconds=round(time.time() - overall_start, 2),
            test_results=results,
        )
