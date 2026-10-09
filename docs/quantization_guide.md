# GGUF Quantization and Calibration Guide

This guide describes the quantization formats supported in llama.cpp and QuantForge, along with best practices for calibration and quality testing.

## Quantization Types Overview

llama.cpp implements multiple quantization families, each designed for different trade-offs between memory footprint, inference speed, and perplexity.

### 1. Legacy Quants

- **Q4_0**: Basic 4-bit uniform quantization. 32-element blocks with 16-bit float scale. Fast computation, but superseded by K-quants in output quality.
- **Q4_1**: 4-bit quantization with scale and minimum offset. Offers slightly higher precision than Q4_0 with minimal memory overhead.
- **Q5_0 / Q5_1**: 5-bit uniform quantization. Higher quality than 4-bit formats with moderate size increase.
- **Q8_0**: 8-bit quantization. Often used as an almost lossless reference format for weights.

### 2. K-Quants (Block-Level Mixed Precision)

K-quants vary the quantization bit depth according to tensor significance. Critical weights (such as attention projections and feed-forward gates) retain higher bit representations.

- **Q2_K**: Extremely aggressive 2-bit quantization with 4-bit scale blocks. Suitable only when severe RAM limits prevent loading larger models.
- **Q3_K_S / Q3_K_M / Q3_K_L**: 3-bit quantization variants. Small (S) uses lower bits across more tensors; Medium (M) balances attention and feed-forward layers; Large (L) protects key projections.
- **Q4_K_S / Q4_K_M**: Standard 4-bit quantization. `Q4_K_M` is the common daily driver recommendation for modern models, offering strong balance between size and response quality.
- **Q5_K_S / Q5_K_M**: 5-bit mixed quantization. Retains high precision with roughly 1 bit extra memory over Q4.
- **Q6_K**: 6-bit quantization. Very close to FP16 fidelity with roughly 50% memory savings compared to unquantized checkpoints.

### 3. Importance Matrix Quants (I-Quants)

I-quants compute an importance matrix (`imatrix`) over a calibration text corpus before quantization. Tensors with high activation variance receive higher precision, minimizing quantization error in sub-4-bit regimes.

- **IQ1_S / IQ1_M**: Sub-2-bit formats (1.5 to 1.7 bits per weight). Require calibration to maintain coherent text output.
- **IQ2_XXS / IQ2_XS / IQ2_S / IQ2_M**: 2-bit variants with varying block allocations. `IQ2_M` provides usable responses in memory-constrained environments where standard Q2 fails.
- **IQ3_XXS / IQ3_S / IQ3_M**: 3-bit importance formats. `IQ3_M` frequently outperforms standard `Q3_K_M` at similar or smaller file sizes.
- **IQ4_XS / IQ4_NL**: Non-linear and compact 4-bit formats. `IQ4_NL` applies non-linear quantization grids to better reflect weight distributions.

## Importance Matrix Calibration

### Preparing Calibration Corpora

Calibration data should match the target deployment domain:

1. **General language**: A sample of diverse Wikipedia text, news articles, or multi-topic essays.
2. **Code**: Repository code snippets across target programming languages.
3. **Conversational**: Multi-turn dialogue transcripts with diverse system prompts.

### Calibration Parameters

- **Context Size (`--ctx-size`)**: Typically 512 to 2048 tokens. Larger contexts capture long-range dependencies but increase calibration time and VRAM usage.
- **Chunk Count (`--chunks`)**: Aim for 32 to 128 chunks. Diminishing returns occur beyond 256 chunks for models under 14B parameters.
- **GPU Offload (`--n-gpu-layers`)**: Offload layers to accelerate calibration if sufficient GPU memory is available.

## Perplexity Evaluation

Perplexity measures how well the model predicts sample text. Lower scores indicate better retention of language modeling capability.

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid w_{<i})\right)$$

### Evaluation Best Practices

- Always evaluate against an independent test set that was excluded from calibration.
- Standard benchmark text: `wikitext-2-raw/wiki.test.raw`.
- Track delta perplexity: $\Delta \text{PPL} = \text{PPL}_{\text{quant}} - \text{PPL}_{\text{FP16}}$. An acceptable degradation for everyday use is typically under $+0.2$ PPL points.
