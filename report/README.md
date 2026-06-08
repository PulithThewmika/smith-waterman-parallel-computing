# `report/` — Figures

Generated performance figures for the project.

| Path | Contents |
| --- | --- |
| [`graphs/`](graphs/) | All performance graphs (PNG) produced by `scripts/plot.py` — see [graphs/README.md](graphs/README.md) |

## Regenerating the figures

```bash
./scripts/run_benchmarks.sh 8000 8000 12345   # -> data/results.csv
python3 scripts/plot.py                         # -> report/graphs/*.png
```

The full numerical results, discussion, and analysis are in the [root README](../README.md).
