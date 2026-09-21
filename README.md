# Protein Ligand Binding Models

Fit native-MS protein-ligand bound-state fractions with sequential specific and nonspecific-adduct models. The Python import is `scripts_binding`.

## Run the real-data example

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
protein-ligand-fit examples/adp_amac_20c/fit.yaml
```

The example fits three ADP/AmAc native-MS replicates at 20 °C. Input CSVs, configuration, provenance, and the expected graphic are together in [examples/adp_amac_20c](examples/adp_amac_20c/README.md). The command writes per-replicate fit SVGs and fitted-constant CSVs, plus a combined figure with observed replicate error bars, a summary CSV, and an optimizer log under `examples/adp_amac_20c/output/`.

![ADP AmAc 20 C combined fit: observed mean bound-state fractions with replicate error bars and mean fitted curves](examples/adp_amac_20c/fit_preview.png)

The preview combines the three independent fits. Error bars show sample SD of measured replicates where at least two are available. The 30 µM point has one replicate and no error bar. This figure is not a claim that the model uniquely identifies binding sites or mechanisms.

## Other example

[examples/synthetic](examples/synthetic/README.md) shows a two-site fitting check and forward simulation from supplied constants. See the [example index](examples/README.md) for both commands.

## Input and output

Titration CSVs use `Entry` for total ligand concentration and `I0`, `I1`, etc. for bound-state intensities or fractions. Set ligand units and protein concentration in YAML. Missing intensity cells remain missing; measured zeros remain zero. Inputs accept UTF-8, UTF-8 BOM, UTF-16 BOM, and Windows-1252. Output CSVs use UTF-8 with comma separators and decimal points.

Model families: sequential specific, sequential adduct, competing adduct, stochastic adduct, occupancy decay, and shared-site. Thermodynamic analysis code is included for temperature series. Check fit quality and parameter identifiability before biological interpretation.

## Build

```bash
python -m pip install build
python -m build
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
