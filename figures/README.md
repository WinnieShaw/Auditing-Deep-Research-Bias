# Paper figures

This directory contains the three figures used in the paper.

| Paper figure | Publication asset | Editable or source artifact |
| --- | --- | --- |
| Figure 1: method overview | `overview.pdf` | `overview_editable.pptx` |
| Figure 2: ranking sensitivity | `fig_search_bias_and_stability.pdf` | `fig_search_bias_and_stability.svg`; `source_data/fig_search_bias_and_stability.csv` |
| Figure 3: budget efficiency | `fig_budget_efficiency.pdf` | `fig_budget_efficiency.svg`; `source_data/fig_budget_efficiency.csv` |

Figures 2 and 3 are reproduced from the released CSV files with:

```bash
python3 scripts/plot_paper_figures.py
```

Use `--source-dir` and `--output-dir` to read or write elsewhere. The script
emits PDF, SVG and PNG. It uses Times New Roman when installed and otherwise
falls back to an available Times-compatible serif font. The checked-in PDFs are
the canonical publication assets; regenerated files can differ at the binary
level because PDF metadata and installed fonts vary across systems.

The controlled panel of Figure 2 reports the unshrunk sequential correction
underlying CESS. Its short plot label is `CESS correction`. The real-task panel
and Figure 3 use task-wise CESS, as shown in their legends.
