# Synthetic two-site example

## Run

From the repository root after installation:

```bash
protein-ligand-fit examples/synthetic/fit.yaml
```

![Synthetic two-site fit](fit_preview.png)

To generate bound-state curves from the supplied constants:

```bash
protein-ligand-simulate --params examples/synthetic/parameters.csv --out-dir examples/synthetic/simulation --p-tot 1 --p-tot-unit uM --ligand-max 160 --ligand-unit uM --csv
```

## Inputs and expected result

| File | Purpose |
|---|---|
| `fit.yaml` | Protein concentration, units, model, and output path |
| `titration.csv` | Six ligand concentrations with three bound states |
| `parameters.csv` | Forward simulation example |

The synthetic titration uses a two-site sequential-specific model with 5 and 20 micromolar dissociation constants. Check the generated fitted-constant CSV, fit SVG, and optimization log under `output_demo/sequential_specific/` for recovery and fit diagnostics.

Data flow: CSV intensities → row fractions → free-ligand mass balance → model fit → fitted constants and diagnostics.
