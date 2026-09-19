"""Перебор landcover-фильтров AF: какие классы исключать как техногенку.

Идея: правило I4>=330 & d45>=25 ловит нагретый асфальт, газовые факелы,
промышленные площадки. Landcover из aux-канала помогает их отсеять.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# AF-правило (зафиксировано grid search)
AF_I4_MIN = 330.0
AF_D45_MIN = 25.0

# Индексы каналов (1-based для rasterio)
BAND_I4 = 4
BAND_I5 = 5
BAND_AUX_LANDCOVER = 1

# Классы ESA WorldCover
LC_FOREST = 10
LC_GRASS = 30
LC_CROPLAND = 40
LC_BUILTUP = 50
LC_BARE = 60
LC_WATER = 80
LC_WETLAND = 90


# (имя, set исключаемых классов)
VARIANTS = [
    ("baseline",                    set()),
    ("no_builtup",                  {LC_BUILTUP}),
    ("no_builtup_bare",             {LC_BUILTUP, LC_BARE}),
    ("no_builtup_cropland",         {LC_BUILTUP, LC_CROPLAND}),
    ("no_builtup_bare_cropland",    {LC_BUILTUP, LC_BARE, LC_CROPLAND}),
    ("only_vegetation",             {LC_BUILTUP, LC_BARE, LC_WATER}),
]


def predict_af_with_filter(viirs_path, aux_path, exclude_classes: set):
    with rasterio.open(viirs_path) as src:
        i4 = src.read(BAND_I4).astype(np.float32)
        i5 = src.read(BAND_I5).astype(np.float32)

    valid = ~np.isnan(i4) & ~np.isnan(i5)
    if not valid.any():
        return np.zeros(i4.shape, dtype=np.uint8)

    i4n = np.nan_to_num(i4, nan=0.0)
    i5n = np.nan_to_num(i5, nan=0.0)

    fire = valid & (i4n >= AF_I4_MIN) & ((i4n - i5n) >= AF_D45_MIN)

    if exclude_classes:
        with rasterio.open(aux_path) as src:
            landcover = src.read(BAND_AUX_LANDCOVER).astype(np.int32)
        # Исключаем пиксели в этих классах
        for lc in exclude_classes:
            fire = fire & (landcover != lc)

    return fire.astype(np.uint8)


def main(data_dir: Path) -> None:
    meta = pd.read_csv(data_dir / "af" / "meta.csv")
    print(f"AF-чипов в train: {len(meta)}\n")

    # --- Загрузка всех чипов один раз (в память) ---
    # ~420 × 256×256 × 3 float32 ≈ 350 МБ — влезает
    cache = []  # (i4, i5, valid, landcover, mask)
    skipped = 0

    for i, row in meta.iterrows():
        cid = row["chip_id"]
        viirs = data_dir / "af" / "viirs" / f"{cid}_VIIRS_I1-I5.tif"
        aux   = data_dir / "af" / "aux"   / f"{cid}_AUX.tif"
        mask_p = data_dir / "af" / "masks" / f"{cid}_MASK.tif"

        if not (viirs.exists() and aux.exists() and mask_p.exists()):
            skipped += 1
            continue

        with rasterio.open(viirs) as src:
            i4 = src.read(BAND_I4).astype(np.float32)
            i5 = src.read(BAND_I5).astype(np.float32)

        with rasterio.open(aux) as src:
            landcover = src.read(BAND_AUX_LANDCOVER).astype(np.int32)

        with rasterio.open(mask_p) as src:
            mask = src.read(1).astype(np.uint8)
            mask[mask == 255] = 0

        valid = ~np.isnan(i4) & ~np.isnan(i5)
        i4n = np.nan_to_num(i4, nan=0.0)
        i5n = np.nan_to_num(i5, nan=0.0)

        cache.append((i4n, i5n, valid, landcover, mask))

        if (i + 1) % 100 == 0:
            print(f"  загрузка {i+1}/{len(meta)}")

    print(f"Загружено: {len(cache)}, пропущено: {skipped}\n")

    # --- Перебор вариантов ---
    print("=" * 82)
    print(f"{'Вариант':<28} {'F1':>8} {'P':>8} {'R':>8} "
          f"{'TP':>7} {'FP':>7} {'FN':>7}")
    print("=" * 82)

    results = []

    for name, exclude in VARIANTS:
        tp = fp = fn = 0
        for i4n, i5n, valid, landcover, mask in cache:
            fire = valid & (i4n >= AF_I4_MIN) & ((i4n - i5n) >= AF_D45_MIN)
            for lc in exclude:
                fire = fire & (landcover != lc)

            tp += int((fire & (mask == 1)).sum())
            fp += int((fire & (mask == 0)).sum())
            fn += int((~fire & (mask == 1)).sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        print(f"{name:<28} {f1:>8.4f} {p:>8.4f} {r:>8.4f} "
              f"{tp:>7} {fp:>7} {fn:>7}")

        results.append(dict(name=name, f1=f1, p=p, r=r,
                            tp=tp, fp=fp, fn=fn))

    # --- Лучший ---
    df = pd.DataFrame(results)
    best = df.loc[df["f1"].idxmax()]
    base = df.iloc[0]

    print()
    print(f"🏆 Лучший: {best['name']}")
    print(f"   F1 = {best['f1']:.4f}  (baseline = {base['f1']:.4f}, "
          f"прирост {best['f1'] - base['f1']:+.4f})")
    print(f"   FP: {int(base['fp'])} → {int(best['fp'])}  "
          f"({(best['fp'] - base['fp']) / base['fp'] * 100:+.1f}%)")
    print(f"   TP: {int(base['tp'])} → {int(best['tp'])}  "
          f"({(best['tp'] - base['tp']) / base['tp'] * 100:+.1f}%)")

    # --- Сохранение ---
    out = Path("data/artifacts/tune_af_filter_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ Сохранено: {out}")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train")
    main(data_dir)