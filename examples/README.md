# Examples

| Folder | Input | Command | Output |
|---|---|---|---|
| [ADP/AmAc at 20 °C](adp_amac_20c/README.md) | Three real native-MS replicate CSVs | `python run.py examples/adp_amac_20c/fit.yaml` | Per-replicate constants and fits, plus a combined figure with observed error bars |
| [Synthetic two-site](synthetic/README.md) | Generated titration and parameter CSV | `python run.py` | Known-parameter fitting check and forward simulation |

Run commands from the repository root after `python -m pip install -e .`. Paths inside fitting YAML files are relative to those files.
