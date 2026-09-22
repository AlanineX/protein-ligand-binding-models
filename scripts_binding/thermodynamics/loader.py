"""Load replicate or uncertainty-weighted Kd temperature-series CSVs."""

import csv
import io
from pathlib import Path

import numpy as np

from scripts_binding.core.csv_io import decode_csv


def _is_uncertainty_header(h):
    """Header looks like an uncertainty column for a Kd column."""
    h = h.strip()
    return h.startswith("e_") and ("uM" in h or "μM" in h)


def _is_kd_header(h):
    """Header looks like a Kd column (not its uncertainty counterpart)."""
    h = h.strip()
    if not h or h.startswith("e_"):
        return False
    return "uM" in h or "μM" in h


def load_kd_csv(filepath):
    """Parse Kd values and optional uncertainty columns by temperature and site."""
    filepath = Path(filepath).resolve()
    warnings = []

    content, _ = decode_csv(filepath)
    with io.StringIO(content, newline="") as f:
        header = next(csv.reader(f))

    # Identify Kd columns (and matching uncertainty columns by paired position)
    kd_cols = []
    se_cols = {}     # Kd column index -> matching e_* column index
    for i, h in enumerate(header):
        if i >= 2 and _is_kd_header(h):
            kd_cols.append(i)
        elif i >= 2 and _is_uncertainty_header(h):
            # Match to the immediately preceding Kd column with the same suffix
            base = h.strip()[2:]   # strip 'e_'
            for kc in kd_cols[::-1]:
                if header[kc].strip() == base:
                    se_cols[kc] = i
                    break

    if not kd_cols:
        raise ValueError(f"No Kd columns found in {filepath}")

    fmt = "weighted" if se_cols else "replicate"
    n_sites = len(kd_cols)
    site_labels = [header[i].strip() for i in kd_cols]

    # Parse data rows
    data_by_temp = {}
    se_by_temp = {}
    rep_labels_by_temp = {}

    with io.StringIO(content, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or not row[0].strip():
                continue
            temp_C = float(row[0].strip())
            rep_label = row[1].strip() if len(row) > 1 else ""

            kd_values = []
            se_values = []
            for ci in kd_cols:
                if ci < len(row) and row[ci].strip():
                    try:
                        val = float(row[ci].strip())
                        kd_values.append(val if val > 0 else np.nan)
                    except ValueError:
                        kd_values.append(np.nan)
                else:
                    kd_values.append(np.nan)

                # Paired uncertainty if present
                ec = se_cols.get(ci)
                if ec is not None and ec < len(row) and row[ec].strip():
                    try:
                        sev = float(row[ec].strip())
                        se_values.append(sev if sev > 0 else np.nan)
                    except ValueError:
                        se_values.append(np.nan)
                else:
                    se_values.append(np.nan)

            data_by_temp.setdefault(temp_C, []).append(np.array(kd_values))
            se_by_temp.setdefault(temp_C, []).append(np.array(se_values))
            rep_labels_by_temp.setdefault(temp_C, []).append(rep_label)

    # Detect duplicate replicates within each temperature (only meaningful
    # under replicate format — weighted format has 1 row per temperature)
    if fmt == "replicate":
        for temp_C in sorted(data_by_temp):
            reps = data_by_temp[temp_C]
            labels = rep_labels_by_temp[temp_C]
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    if np.allclose(reps[i], reps[j], equal_nan=True):
                        warnings.append(
                            f"T={temp_C}C: replicates '{labels[i]}' and "
                            f"'{labels[j]}' are identical (duplicate data)"
                        )

    return {
        "temperatures": sorted(data_by_temp.keys()),
        "n_sites": n_sites,
        "kd_data": data_by_temp,
        "kd_se_data": se_by_temp,
        "format": fmt,
        "site_labels": site_labels,
        "filepath": filepath,
        "warnings": warnings,
    }
