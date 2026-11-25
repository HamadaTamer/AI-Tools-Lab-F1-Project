# LLM comparison

Metrics are defined in `eval/eval_llm.py`. Blank = not applicable.

## Prompt variant: base

| model | style | n | failed | correct | faithful | refusal | expl_recall | expl_no_foreign | expl_fidelity | latency_s | input_tokens | output_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| allam-2-7b | explain | 9 | 0 |  |  |  | 0.667 | 1.0 | 0.667 | 5.0 | 632.78 | 397.33 |
| allam-2-7b | few-shot | 52 | 0 | 0.577 | 0.615 | 0.6 |  |  |  | 5.8 | 1222.9 | 86.63 |
| allam-2-7b | zero-shot | 52 | 0 | 0.776 | 0.404 | 1.0 |  |  |  | 2.12 | 536.9 | 124.75 |
| openai/gpt-oss-20b | explain | 9 | 0 |  |  |  | 0.333 | 1.0 | 0.333 | 1.12 | 588.67 | 205.44 |
| openai/gpt-oss-20b | few-shot | 52 | 0 | 0.782 | 0.962 | 1.0 |  |  |  | 0.99 | 1094.54 | 79.96 |
| openai/gpt-oss-20b | zero-shot | 52 | 0 | 0.79 | 0.981 | 0.4 |  |  |  | 0.94 | 501.54 | 91.33 |
| qwen/qwen3.8-27b | explain | 9 | 0 |  |  |  | 0.296 | 1.0 | 0.296 | 1.24 | 581 | 163.44 |
| qwen/qwen3.8-27b | few-shot | 51 | 0 | 0.724 | 1.0 | 1.0 |  |  |  | 1.03 | 1151.98 | 50.8 |
| qwen/qwen3.8-27b | zero-shot | 52 | 0 | 0.74 | 1.0 | 1.0 |  |  |  | 1.35 | 490.31 | 48.48 |

### Few-shot minus zero-shot (base)

| model | correct | faithful | refusal | latency_s | input_tokens |
| --- | --- | --- | --- | --- | --- |
| allam-2-7b | -0.199 | +0.211 | -0.400 | +3.680 | +686.000 |
| openai/gpt-oss-20b | -0.008 | -0.019 | +0.600 | +0.050 | +593.000 |
| qwen/qwen3.8-27b | -0.016 | +0.000 | +0.000 | -0.320 | +661.670 |

## Prompt variant: strict

| model | style | n | failed | correct | faithful | refusal | expl_recall | expl_no_foreign | expl_fidelity | latency_s | input_tokens | output_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| allam-2-7b | explain | 9 | 0 |  |  |  | 0.741 | 1.0 | 0.741 | 6.67 | 739.56 | 389.78 |
| allam-2-7b | few-shot | 52 | 0 | 0.424 | 0.75 | 1.0 |  |  |  | 9.33 | 1462.9 | 89.83 |
| allam-2-7b | zero-shot | 52 | 0 | 0.834 | 0.442 | 0.8 |  |  |  | 2.33 | 596.9 | 111.62 |
| openai/gpt-oss-20b | explain | 9 | 0 |  |  |  | 0.963 | 1.0 | 0.963 | 0.94 | 716.22 | 191.22 |
| openai/gpt-oss-20b | few-shot | 52 | 0 | 0.763 | 0.962 | 1.0 |  |  |  | 0.82 | 1318.54 | 83.42 |
| openai/gpt-oss-20b | zero-shot | 52 | 0 | 0.79 | 1.0 | 0.7 |  |  |  | 0.84 | 557.54 | 89.46 |
| qwen/qwen3.8-27b | explain | 9 | 0 |  |  |  | 0.963 | 1.0 | 0.963 | 0.98 | 702.56 | 164.22 |
| qwen/qwen3.8-27b | few-shot | 52 | 2 | 0.747 | 1.0 | 1.0 |  |  |  | 0.87 | 1389.62 | 49.92 |
| qwen/qwen3.8-27b | zero-shot | 52 | 1 | 0.813 | 1.0 | 1.0 |  |  |  | 2.31 | 551.51 | 48.24 |

### Few-shot minus zero-shot (strict)

| model | correct | faithful | refusal | latency_s | input_tokens |
| --- | --- | --- | --- | --- | --- |
| allam-2-7b | -0.410 | +0.308 | +0.200 | +7.000 | +866.000 |
| openai/gpt-oss-20b | -0.027 | -0.038 | +0.300 | -0.020 | +761.000 |
| qwen/qwen3.8-27b | -0.066 | +0.000 | +0.000 | -1.440 | +838.110 |
