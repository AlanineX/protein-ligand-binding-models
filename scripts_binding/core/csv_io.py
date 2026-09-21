"""Read common CSV encodings without silently replacing undecodable bytes."""

from pathlib import Path
import io

import pandas as pd


def decode_csv(path):
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode CSV: {path}")


def read_csv(path, **kwargs):
    content, _ = decode_csv(path)
    return pd.read_csv(io.StringIO(content), **kwargs)
