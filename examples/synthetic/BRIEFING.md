# Synthetic demo briefing

## Run

From the repository root after installation:

```bash
protein-ligand-fit examples/synthetic/fit.yaml
```

## Inputs and expected result

| File | Purpose |
|---|---|
| `fit.yaml` | Protein concentration, units, model, and output path |
| `titration.csv` | Nine ligand concentrations with three bound states |
| `parameters.csv` | Forward simulation example |

The synthetic titration uses a two-site sequential-specific model with 5 and 20 micromolar dissociation constants. Check the generated fitted-constant CSV and optimization log under `output_demo/` for recovery and fit diagnostics.

Data flow: CSV intensities → row fractions → free-ligand mass balance → model fit → fitted constants and diagnostics.
