# Percolative Chaos-Lifting Network (PCLN)

**Sub-quadratic Transformer Attention via Percolation Theory on Random Geometric Graphs**

* **📖 Article [RU]:** [Theory and Implementation Details](https://synthcore.org/perkoljacionnoe-vnimanie/)
* **📺 Video Explanation [RU]:**
[![Percolation Attention](https://img.youtube.com/vi/JkSxlxzeN7w/maxresdefault.jpg)](https://youtu.be/JkSxlxzeN7w)

[![Status](https://img.shields.io/badge/status-experimental-orange)]()

## Abstract

Modern Transformers suffer from $O(N^2)$ attention complexity and flat Euclidean feature representations. PCLN addresses both by replacing dense pairwise attention with **dynamic message-passing on percolation graphs** — a principled sparse mechanism that achieves $O(N \log N)$ complexity with mathematical guarantees of global context preservation.

## Key Results

| Metric | Baseline (Full Attention) | PCLN (Message-Passing) | Improvement |
|--------|--------------------------|------------------------|-------------|
| **cos_sim** (synthetic) | 1.00 | **0.954** at 99.98% sparsity | 0.02% connections |
| **cos_sim** (synthetic) | 0.003 (masked) | **0.954** (msg-pass) | +0.95 |
| **PPL** (Qwen-0.5B, frozen) | 5.55 | **5.61** (+1.0%) | 94% sparse attention |
| **Latency** (2048 tokens) | 73ms | **8ms** | 9× faster |
| **Scaling** | $O(N^2)$ | **$O(N \log N)$** | Confirmed |

## Architecture

PCLN synthesizes four mathematical frameworks from 94 arXiv papers:

1. **Sub-Lorentzian Geometry** — token projection into Heisenberg group (causality-embedded metric, no Positional Encoding) — `[2605.31397v1]`
2. **Percolative Attention** — Random Geometric Graph with Simon-Lieb inequality; signal flows only through percolation clusters — `[2605.30299v1]`, `[2606.01627v1]`
3. **PCE Weight Generator** — weights as Galerkin polynomial expansions instead of static matrices — `[2605.31288v1]`
4. **MCJAC + Accelerated Sinkhorn** — Hessian supervision + Nesterov-accelerated Optimal Transport — `[2606.01596v1]`, `[2605.30267v2]`

**Status:** Blocks 1, 3, 4 — theoretical only. **Block 2 — experimentally validated** (Stages 1-3).

## Quick Start

```bash
# Stage 1: Percolation threshold & scaling test
python experiments/percolative_attention/test_percolation.py

# Stage 2: Message-passing vs masked attention (synthetic)
python experiments/percolative_attention/test_message_passing.py

# Stage 3: Frozen LLM test (Qwen-0.5B)
python experiments/percolative_attention/test_frozen_llm.py
```

**Requirements:** `torch`, `numpy`, `scipy`, `transformers`

## Project Structure

```
experiments/
├── percolative_attention/
│   ├── test_percolation.py         # Stage 1: Δ_c, scaling law
│   ├── test_message_passing.py     # Stage 2: MsgPass vs Masked
│   ├── test_frozen_llm.py          # Stage 3: GPT-2 / Qwen test
│   ├── README.md                   # Full architecture description
│   └── test_results_*.json         # Raw experiment data
├── cpkn/
│   ├── cpkn_smoketest.py           # CPKN 3-concept smoke test
│   ├── test_kronecker_lora.py      # LoRA vs Structured LoRA
│   └── test_gen_compare.py         # Generation quality comparison
├── EXPERIMENTS.csv                 # All experiment records
└── TEST_RESULTS.md                 # Human-readable results
```

## Citation / Sources

Key arXiv papers this work builds upon:

- Simon-Lieb inequality for percolation: `arXiv:2605.30299v1`
- Giant component in RGG: `arXiv:2606.01627v1`
- Sub-Lorentzian Heisenberg group: `arXiv:2605.31397v1`
- Polynomial Chaos Expansion: `arXiv:2605.31288v1`
- MCJAC second-order supervision: `arXiv:2606.01596v1`
- Accelerated Sinkhorn: `arXiv:2605.30267v2`

Full mathematics in `articles/pcln_full_paper.md`.

## License

MIT
