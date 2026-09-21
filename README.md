# Protein Ligand Binding Models

Fit protein-ligand bound-state fractions from native mass spectrometry titrations. The first release provides six model families, forward simulation, plots, and a synthetic fitting demo. The Python import remains `scripts_binding` for compatibility with existing analyses.

## Quick start

Python 3.10 or newer is required. From this repository directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
protein-ligand-fit examples/synthetic/fit.yaml
```

The demo reads `examples/synthetic/titration.csv` and writes fitted constants and a log under `examples/synthetic/output_demo/`. All config paths are relative to the YAML file, so the command also works from another directory. The two-site synthetic data were generated with dissociation constants of 5 and 20 micromolar; fitted values should be close to those numbers.

### Demo fit output

![Synthetic two-site binding fit: measured fractions shown as points and fitted curves as lines](docs/images/synthetic_two_site_fit.png)

The fitting command generates `examples/synthetic/output_demo/sequential_specific/titration_sequential_specific_fit.svg`. The preview above was rendered from that SVG. Points are the bundled synthetic data; lines are the fitted model.

To generate a curve from known parameters:

```bash
protein-ligand-simulate --params examples/synthetic/parameters.csv --out-dir examples/synthetic/simulation --p-tot 1 --p-tot-unit uM --ligand-max 160 --ligand-unit uM --csv
```

## Input and output

Titration CSVs need an `Entry` column for total ligand concentration and `I0`, `I1`, etc. for bound-state intensities or fractions. Set concentration units and protein concentration in YAML. Missing intensity cells stay missing; measured zeros stay zero. UTF-8, UTF-8 with BOM, UTF-16 with BOM, and Windows-1252 CSV files are accepted. Output CSVs use UTF-8 and comma separators. Use a dot as the decimal separator.

The demo config is a starting point for your own data. The output contains fitted dissociation constants, optimization logs, and optional figures. Fit quality and parameter identifiability must be checked before biological interpretation; a good curve fit alone does not prove a binding mechanism.

## Models and scope

The registry includes sequential specific, sequential adduct, competing adduct, stochastic adduct, occupancy decay, and shared-site models. Model selection depends on the experiment and is not automated biological validation. Thermodynamic analysis is available for temperature series, but the synthetic demo covers titration fitting only.

## Development

```bash
python -m pytest -q
python -m build
```

See `examples/synthetic/BRIEFING.md` for the short demo walkthrough. Real research data, publisher PDFs, and generated bulk figures are excluded from this release package.

## License

BSD 3-Clause. See `LICENSE`.
