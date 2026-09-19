"""Инспекция одного BS-чипа: S2 pre/post, S1 pre/post, mask, aux."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio


def describe(path: Path) -> None:
    if not path.exists():
        print(f"❌ Нет файла: {path}")
        return
    with rasterio.open(path) as src:
        print(f"\n📄 {path.name}")
        print(f"   size: {src.width}x{src.height}, bands: {src.count}, dtype: {src.dtypes[0]}")
        print(f"   nodata: {src.nodata}, crs: {src.crs}")
        for b in range(1, src.count + 1):
            arr = src.read(b)
            valid = arr[arr != (src.nodata if src.nodata is not None else np.nan)]
            if valid.size == 0:
                print(f"   band {b}: all nodata")
                continue
            print(f"   band {b}: min={valid.min():.3f}, max={valid.max():.3f}, "
                  f"mean={valid.mean():.3f}, nan={np.isnan(arr).sum()}")


def main(chip_id: str, root: Path) -> None:
    print(f"🔍 Инспекция BS-чипа: {chip_id}\n" + "=" * 60)
    for sub, pat in (
        ("sentinel2_pre", f"{chip_id}_Sentinel-2_pre.tif"),
        ("sentinel2_post", f"{chip_id}_Sentinel-2_post.tif"),
        ("sentinel1_pre", f"{chip_id}_Sentinel-1_pre.tif"),
        ("sentinel1_post", f"{chip_id}_Sentinel-1_post.tif"),
        ("masks", f"{chip_id}_MASK.tif"),
        ("aux", f"{chip_id}_AUX.tif"),
    ):
        describe(root / sub / pat)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train/bs")
    chip = sys.argv[2] if len(sys.argv) > 2 else "BS_tr_000001"
    main(chip, root)