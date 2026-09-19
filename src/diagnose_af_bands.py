"""Диагностика: NaN по каналам для нескольких AF-чипов."""

from pathlib import Path
import numpy as np
import rasterio


def diag(chip_id: str, root: Path = Path("data/train/af/viirs")):
    path = root / f"{chip_id}_VIIRS_I1-I5.tif"
    print(f"\n{'=' * 60}\n{chip_id}")
    with rasterio.open(path) as src:
        print(f"  count={src.count}, dtypes={src.dtypes}, nodata={src.nodatavals}")
        for i in range(1, src.count + 1):
            arr = src.read(i)
            n_nan = int(np.isnan(arr).sum())
            finite = arr[~np.isnan(arr)]
            if finite.size > 0:
                fmin, fmax = float(finite.min()), float(finite.max())
                n_zeros = int((arr == 0).sum())
                n_neg9999 = int((arr == -9999).sum())
                n_neg32768 = int((arr == -32768).sum())
            else:
                fmin = fmax = float("nan")
                n_zeros = n_neg9999 = n_neg32768 = 0
            print(f"  band {i}: nan={n_nan:>6}/{arr.size} "
                  f"({n_nan/arr.size:.3f})  min={fmin:.3f} max={fmax:.3f}  "
                  f"zeros={n_zeros}  -9999={n_neg9999}  -32768={n_neg32768}")


if __name__ == "__main__":
    # Один чистый, два "полных NaN", один частичный
    for chip in ["AF_tr_000001", "AF_tr_000033", "AF_tr_000037",
                 "AF_tr_000263", "AF_tr_000032"]:
        diag(chip)