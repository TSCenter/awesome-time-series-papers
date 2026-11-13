# Conference Papers Update — 2026/11/13

Added two previously missing main-conference papers, using Asia/Shanghai (Beijing time).

## Time Series Forecasting

| Method Name | Code | Source |
| ------- | ------ | ---------- |
| [MetaCaster: Meta-Harness-Optimized Agent for End-to-End Few-Shot Learning of Lightweight Time Series Forecasters](https://arxiv.org/abs/2608.23473) | [Code](https://github.com/D2I-Group/metacaster) | EMNLP 2026 |
| [What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions](https://arxiv.org/abs/2605.14422) | [Code](https://github.com/seqml/TADiff) | ICML 2026 |

## Source verification

- **MetaCaster:** [EMNLP main-conference proceedings](https://aclanthology.org/2026.emnlp-main.2511/) match the title and all nine authors of arXiv:2608.23473v2. The [paper](https://arxiv.org/html/2608.23473v2) explicitly links the implementation repository, whose README identifies this paper and documents its agent entry points.
- **TADiff:** [ICML proceedings](https://proceedings.mlr.press/v306/gu26r.html) match the title and all four authors of arXiv:2605.14422v1. The arXiv-linked [author project page](https://seqml.github.io/TADiff/) links the implementation repository, which identifies this exact paper and includes model, training, and evaluation code.

Deduplicated against the complete README by title, alias, arXiv identifier, and code URL. Existing paper rows and categories were preserved. Replaced only the prior EMNLP 2026 and ICML 2026 update notices.

Deferred candidates: OliO and Echoes of Error appear in the official NeurIPS download directory, but their specific track records could not be retrieved; TimeSense has a title/author discrepancy with the located arXiv version; no matching arXiv record was located for TiMER. AlphaCast and Perseus require independently readable main-track proceedings or acceptance evidence. These candidates were not added.
