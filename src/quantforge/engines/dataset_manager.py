"""Calibration dataset presets and corpus preparation manager for llama-imatrix."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from quantforge.core.config import settings


@dataclass
class DatasetPreset:
    """Descriptor for a built-in calibration dataset preset."""

    name: str
    description: str
    category: str
    sample_text: str


PRESET_WIKI = """In mathematics and computer science, an algorithm is a finite sequence of rigorous instructions, typically used to solve a class of specific problems or to perform a computation. Algorithms are used as specifications for performing calculations, data processing, automated reasoning, automated decision-making and other tasks. In contrast, a heuristic is an approach to problem solving that may not be fully specified or may not guarantee correct or optimal results, especially in problem domains where there is no well-defined correct or optimal result.

As an effective method, an algorithm can be expressed within a finite amount of space and time, and in a well-defined formal language for calculating a function. Starting from an initial state and initial input (perhaps empty), the instructions describe a computation that, when executed, proceeds through a finite number of well-defined successive states, eventually producing output and terminating at a final ending state. The transition from one state to the next is not necessarily deterministic; some algorithms, known as randomized algorithms, incorporate random input.

The concept of algorithm has existed for centuries. Partial formalizations of what would become the modern concept of algorithm began with attempts to resolve the Entscheidungsproblem (decision problem) posed by David Hilbert in 1928. Later formalizations were framed as attempts to define effective calculability or effective method. Those formalizations included the Gödel-Herbrand-Kleene recursive functions of 1930, 1934 and 1935, Alonzo Church's lambda calculus of 1936, Emil Post's Formulation 1 of 1936, and Alan Turing's Turing machines of 1936–1937 and 1939."""

PRESET_CODE = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)

async function fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return { id: data.id, name: data.username, active: data.is_active };
}"""

PRESET_MATH = """Problem: Find the maximum value of f(x) = -2x^2 + 8x + 5.
Solution:
1. Since the coefficient of x^2 is -2 (which is negative), the parabola opens downward, so the vertex represents the maximum.
2. The x-coordinate of the vertex for quadratic function ax^2 + bx + c is given by x = -b / (2a).
3. Here, a = -2 and b = 8.
   x = -8 / (2 * -2) = -8 / -4 = 2.
4. Now, substitute x = 2 into f(x) to compute the maximum value:
   f(2) = -2(2)^2 + 8(2) + 5
   f(2) = -2(4) + 16 + 5
   f(2) = -8 + 16 + 5 = 13.
Therefore, the maximum value of f(x) is 13, attained at x = 2.

Theorem (Euler's Identity): For any real number x, e^(i*x) = cos(x) + i*sin(x).
Setting x = pi yields:
e^(i*pi) = cos(pi) + i*sin(pi) = -1 + i(0) = -1.
Hence, e^(i*pi) + 1 = 0, connecting five fundamental mathematical constants."""

PRESET_MULTILINGUAL = """English: Machine learning models require careful quantization to retain linguistic nuances and reasoning capabilities under tight memory constraints.
Romanian: Modelele de inteligență artificială cuantificate prin matrice de importanță mențin un nivel ridicat de acuratețe și o latență redusă pe hardware local.
French: Les techniques de quantification avancées permettent d'exécuter des modèles de langage volumineux directement sur des appareils personnels.
German: Die Optimierung von Sprachmodellen durch Präzisionsanpassung ermöglicht effiziente Inferenz auf handelsüblicher Hardware.
Spanish: La calibración de matrices de importancia preserva la coherencia del razonamiento en modelos de parámetros reducidos."""


class DatasetManager:
    """Manages calibration corpora for llama-imatrix."""

    PRESETS: Dict[str, DatasetPreset] = {
        "general-wiki": DatasetPreset(
            name="general-wiki",
            description="Encyclopedic knowledge, science, and history passages.",
            category="general",
            sample_text=PRESET_WIKI,
        ),
        "code-multilang": DatasetPreset(
            name="code-multilang",
            description="Algorithmic code patterns in Python and JavaScript.",
            category="code",
            sample_text=PRESET_CODE,
        ),
        "reasoning-math": DatasetPreset(
            name="reasoning-math",
            description="Mathematical derivations, proofs, and multi-step logic.",
            category="reasoning",
            sample_text=PRESET_MATH,
        ),
        "multilingual-mixed": DatasetPreset(
            name="multilingual-mixed",
            description="Multilingual benchmark sentences across English, Romanian, French, German, and Spanish.",
            category="multilingual",
            sample_text=PRESET_MULTILINGUAL,
        ),
    }

    def list_presets(self) -> List[DatasetPreset]:
        """Returns all available dataset presets."""
        return list(self.PRESETS.values())

    def get_preset(self, name: str) -> DatasetPreset:
        """Retrieves a specific preset by key."""
        clean = name.strip().lower()
        if clean not in self.PRESETS:
            raise KeyError(f"Unknown dataset preset '{name}'. Available: {list(self.PRESETS.keys())}")
        return self.PRESETS[clean]

    def generate_corpus(
        self,
        preset_name: str,
        target_path: Optional[Path] = None,
        repeats: int = 30,
    ) -> Path:
        """Generates a calibration corpus file from a preset."""
        preset = self.get_preset(preset_name)
        dest = (target_path or settings.cache_path / f"calibration_{preset.name}.txt").resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        full_text = "\n\n".join([preset.sample_text] * repeats)
        dest.write_text(full_text, encoding="utf-8")
        return dest

    def prepare_corpus(
        self,
        input_files: List[Path],
        output_path: Path,
        deduplicate: bool = True,
    ) -> Path:
        """Combines multiple raw text files into a clean calibration corpus."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        seen_blocks = set()
        cleaned_blocks = []

        for f_path in input_files:
            p = Path(f_path).resolve()
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            # Split by double newline paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            for para in paragraphs:
                if deduplicate:
                    if para in seen_blocks:
                        continue
                    seen_blocks.add(para)
                cleaned_blocks.append(para)

        out.write_text("\n\n".join(cleaned_blocks), encoding="utf-8")
        return out
