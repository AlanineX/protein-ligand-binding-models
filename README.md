# Protein Ligand Binding Models

Fit native-MS protein-ligand bound-state fractions with sequential specific and nonspecific-adduct models. The Python import is `scripts_binding`.

## Run the real-data example

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python run.py
```

In Windows VS Code, open the repository folder, select the `.venv` Python interpreter, open `run.py`, and click **Run Python File**. It runs the ADP/AmAc example by default. To choose a different YAML for that button, edit just the `YAML_FILE = "examples/adp_amac_20c/fit.yaml"` line in `run.py`. Paths are relative to the repository root.

| Path setting | Relative to |
|---|---|
| `YAML_FILE` in `run.py` | Repository root |
| `base_dir`, `data_path`, `deconv_csv_path` in YAML | YAML file's directory |
| `csv_pattern`, `output_folder` in YAML | Resolved `base_dir` |

Absolute paths are accepted for `YAML_FILE`, `base_dir`, `data_path`, and `deconv_csv_path`. Use forward slashes in both `run.py` and YAML on Windows, such as `C:/data/titration.csv`; they avoid backslash escapes in quoted strings.

`data_path` selects one exact CSV. If it is unset, `csv_pattern` matches one or more CSVs; `*` is a wildcard, and `{t}` expands from `temperatures`. `output_folder` names the results folder and can also use `{t}`. Existing YAMLs using `wildcard_fmt` and `out_fmt` remain accepted.

The example fits three ADP/AmAc native-MS replicates at 20 °C. Its CSVs, YAML, and preview are together under `examples/adp_amac_20c/`. The command writes per-replicate fit SVGs and fitted-constant CSVs, plus a combined figure with observed replicate error bars, a summary CSV, and an optimizer log under `examples/adp_amac_20c/output/`.

![ADP AmAc 20 C combined fit: observed mean bound-state fractions with replicate error bars and mean fitted curves](examples/adp_amac_20c/fit_preview.png)

The preview combines the three independent fits. Error bars show sample SD of measured replicates where at least two are available. The 30 µM point has one replicate and no error bar. This figure is not a claim that the model uniquely identifies binding sites or mechanisms.

### Real-example settings and provenance

| Item | Value |
|---|---|
| Protein | SR, 1 µM total |
| Ligand and buffer | ADP in AmAc, 20 °C |
| Model | Sequential adduct, 7 specific steps and 2 additional adduct slots |
| Replicates measured | 0, 5, 10, 50 µM: 3; 20, 100 µM: 2; 30 µM: 1 |

The CSVs were copied byte-for-byte from the local research dataset. SHA-256: replicate 1 `95e9a71db642eee6204088e141ed5b11f925522008d1a5a800d7e39241f06ba0`; replicate 2 `320d111958cc38436a792a41a9a3a5b92ac39bb8c0215ecf683f88b1a8f418ab`; replicate 3 `40e03bfa30b9559ce3a69a0d46548729bd4adad51e314b1d1af924c64f93bbff`.

## Synthetic fitting example

```bash
python run.py examples/synthetic/fit.yaml
```

![Synthetic two-site fit](examples/synthetic/fit_preview.png)

This fits the bundled two-site titration generated with 5 and 20 µM dissociation constants. Its heavily commented `fit.yaml` lists every supported fitting setting.

## Input and output

Titration CSVs use `Entry` for total ligand concentration and `I0`, `I1`, etc. for bound-state intensities or fractions. Set ligand units and protein concentration in YAML. Missing intensity cells remain missing; measured zeros remain zero. Inputs accept UTF-8, UTF-8 BOM, UTF-16 BOM, and Windows-1252. Output CSVs use UTF-8 with comma separators and decimal points.

## Models

| Model ID | Binding model | Fitted parameters |
|---|---|---|
| `sequential_specific` | Specific binding steps only | One for each specific step |
| `sequential_adduct` | Stepwise specific binding plus geometric nonspecific adducts | One for each specific step, plus one nonspecific |
| `competing_adduct` | Specific and constant nonspecific contributions to each apparent step | One for each specific step, plus one nonspecific amplitude |
| `stochastic_adduct` | Stepwise specific binding plus Poisson weighted nonspecific adducts | One for each specific step, plus one nonspecific |
| `occupancy_decay` | Stepwise specific binding plus an occupancy-dependent nonspecific contribution | One for each specific step, plus a nonspecific amplitude and shape exponent |
| `shared_site` | Equivalent specific sites sharing one affinity, plus Poisson weighted nonspecific adducts | One shared specific and one nonspecific |

Thermodynamic analysis code is included for temperature series. Check fit quality and parameter identifiability before biological interpretation.

The shared-site model has $n_{\mathrm{params}}(S)=2$ because it fits one shared specific affinity and one nonspecific affinity. $S$ controls how many equivalent specific sites enter the combinatorial model; it does not create one fitted affinity per site. All models retain the same `n_params` interface so the generic fitter can query them uniformly.

### Choose a site count

Use the generic `sequential_specific` model for a new system. Set `s` to its proposed number of specific binding steps and `s_mode: manual` to fit that count. With `s_mode: auto`, the specific-only model uses the highest observed bound-state index; adduct models scan site counts by BIC. In manual mode, adduct models use `s` and infer additional slots from the CSV width unless `n_override` is set. Example:

```yaml
defaults:
  models: [sequential_specific]
  s: 5
  s_mode: manual
```

### Plot colors

Set these YAML keys to Matplotlib colormap names such as `viridis`, `Greens`, or `Purples`:

| Key | Plot elements |
|---|---|
| `colormap` | All apparent bound-state curves; default for deconvolution components |
| `specific_colormap` | Specific-only model curves and pure-specific deconvolution components |
| `nonspecific_colormap` | Deconvolution components containing nonspecific binding |

The default is `colormap: PRGn`; the AmAc example explicitly uses `Greens`. To use one map for everything, set only `colormap`. To use separate component maps, enable `deconv_enable: true` and set the two optional keys. Apparent peaks in adduct models can contain both binding types, so their fit curves use the overall `colormap`.

## Build

```bash
python -m pip install build
python -m build
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
