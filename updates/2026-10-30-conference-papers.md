# Conference Papers Update — 2026/10/30

Added three previously missing main-conference papers: AAAI 2026 (2) and EMNLP 2026 (1). Update date uses Asia/Shanghai (Beijing time).

## Time Series Forecasting

| Method Name | Code | Source |
| ------- | ------ | ---------- |
| [XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](https://arxiv.org/abs/2601.09237) | [Code](https://github.com/Zaiwen/XLinear) | AAAI 2026 |
| [Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](https://arxiv.org/abs/2511.08229) | [Code](https://github.com/decisionintelligence/DTAF) | AAAI 2026 |

## Time Series Analysis and Systems

| Method Name | Code | Source |
| ------- | ------ | ---------- |
| [Grammar of the Wave: Towards Explainable Multivariate Time Series Event Detection via Neuro-Symbolic VLM Agents](https://arxiv.org/abs/2603.11479) | [Code](https://github.com/cw-wan/SELA) | EMNLP 2026 |

## Source verification

- **XLinear:** The [official AAAI proceedings record](https://ojs.aaai.org/index.php/AAAI/article/view/39121) matches the title and all four authors. The arXiv record includes the proceedings DOI; the paper and repository link to each other. Implementation: `models/XLinear.py`.
- **DTAF:** The [official AAAI proceedings record](https://ojs.aaai.org/index.php/AAAI/article/view/39585) matches the title and all six authors. The arXiv record confirms acceptance, and the paper and repository link to each other. Implementation: `ts_benchmark/baselines/dtaf/model/DTAF_model.py`.
- **SELA:** The [ACL Anthology main-conference record](https://aclanthology.org/2026.emnlp-main.290/) matches the title and five authors (Yifei Wang / Yifei Y. Wang differs only by a middle initial). The arXiv v5 record identifies the main-conference camera-ready version; reciprocal repository links and implementation were verified.

Deduplicated against the complete current README using titles, method aliases, arXiv identifiers, and repository URLs. Existing paper entries were preserved.
