# BlendIn: Inference-Time Alignment via Distribution Blending

This is the official implementation of **To Intervene or Not: Guiding Inference-time Alignment with Probabilistic
Model Blending**, accepted at **ACL 2026**.

If you find our work useful, please cite our paper:
```
@inproceedings{blendin2026,
    title     = {To Intervene or Not: Guiding Inference-time Alignment with Probabilistic
Model Blending},
    author    = {Jin Gan, Xin Li∗, Jun Luo},
    booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
    year      = {2026}
}
```
Thank you. Wish you the best. :)

## Setup

```bash
conda create --name basedin python=3.11
pip install vllm==0.6.2
pip install datasets scipy matplotlib seaborn tqdm
export OPENAI_API_KEY="your-key-here"
```

## Hosting Models (vLLM)

```bash
# Base model (port 8000)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --max-logprobs 100 --port 8000 --max_model_len 2048

# Guidance model (port 8001)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-logprobs 100 --port 8001 --max_model_len 2048
```

## Running Experiments

**Base model only (baseline):**
```bash
python run_api.py --dataset_name gsm8k --num_samples 1319 \
    --exp base_only \
    --base_model meta-llama/Llama-3.1-8B \
    --base_host http://localhost:8000/v1 \
    --num_threads 20 --rerun
```

**Guidance model only (baseline):**
```bash
python run_api.py --dataset_name gsm8k --num_samples 1319 \
    --exp nudging_only \
    --nudging_model meta-llama/Llama-3.2-1B-Instruct \
    --nudging_host http://localhost:8001/v1 \
    --num_threads 20 --rerun
```

**BlendIn (distribution blending):**
```bash
python run_api.py --dataset_name gsm8k --num_samples 1319 \
    --exp nudging \
    --base_model meta-llama/Llama-3.1-8B \
    --base_host http://localhost:8000/v1 \
    --nudging_model meta-llama/Llama-3.2-1B-Instruct \
    --nudging_host http://localhost:8001/v1 \
    --top_prob_thres 0.4 \
    --enable_distribution_blending --blend_alpha auto \
    --num_threads 20 --rerun
```

**Original nudging (token-level):**
```bash
python run_api.py --dataset_name gsm8k --num_samples 1319 \
    --exp nudging \
    --base_model meta-llama/Llama-3.1-8B \
    --base_host http://localhost:8000/v1 \
    --nudging_model meta-llama/Llama-3.2-1B-Instruct \
    --nudging_host http://localhost:8001/v1 \
    --top_prob_thres 0.4 \
    --num_threads 20 --rerun
```

## Key Arguments

| Argument | Description |
|---|---|
| `--top_prob_thres` | Uncertainty threshold γ for triggering guidance (default: 0.4) |
| `--enable_distribution_blending` | Use BlendIn distribution blending instead of token-level nudging |
| `--blend_alpha` | Blend weight α (0–1, or `auto` for adaptive) |
| `--num_threads` | Parallel inference threads |
| `--verify_overlap` | Log token distribution overlap statistics at each intervention point |

## Vocabulary Overlap Analysis

```bash
python vocab_analysis.py
```
Results are cached to `vocab_analysis_results.json` after the first run.

## Supported Datasets
`gsm8k`, `mmlu`, `truthfulqa`, `arc_challenge`, `xstest`, `justeval`, `justeval_safe`
