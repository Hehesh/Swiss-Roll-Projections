# Swiss Roll Projections

This project compares linear, randomized, and manifold-aware dimensionality-reduction methods on lifted Swiss-roll classification tasks.

## Refactored Layout

```text
Project/
  README.md
  manifold_project.ipynb
  run_experiments.py
  src/
    manifold_analysis/
      config.py
      data.py
      experiments.py
      metrics.py
      methods.py
      plotting.py
  results/
    figures/
    tables/
```

The notebook can still serve as the narrative report, but the implementation is now split into reusable modules:

- `config.py`: shared experiment settings and label maps
- `data.py`: synthetic Swiss-roll generation and lifting into the ambient space
- `methods.py`: dimensionality-reduction models and transform helpers
- `metrics.py`: neighborhood-preservation scoring
- `experiments.py`: experiment loop plus result summarization
- `plotting.py`: figure generation and file output helpers

## Run

From the `Project/` directory:

```bash
python3 run_experiments.py
```

This regenerates the CSV tables in `results/tables/` and the figures in `results/figures/`.
