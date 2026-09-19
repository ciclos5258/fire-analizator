"""Сверка диагностики NaN с meta.csv: теряем ли мы пиксели горения."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(meta_path: Path, integrity_path: Path) -> None:
    meta = pd.read_csv(meta_path)
    integ = pd.read_csv(integrity_path)

    df = meta.merge(integ, on="chip_id", how="inner")
    print(f"Смержено: {len(df)} из {len(meta)} (meta) и {len(integ)} (integrity)\n")

    # --- Группы по доле NaN ---
    df["nan_group"] = "partial"
    df.loc[df["nan_total"] == 0, "nan_group"] = "clean"
    df.loc[df["nan_total"] == 256 * 256, "nan_group"] = "full_nan"

    print("=" * 70)
    print("РАСПРЕДЕЛЕНИЕ n_fire_px ПО ГРУППАМ NaN")
    print("=" * 70)
    summary = df.groupby("nan_group").agg(
        n_chips=("chip_id", "count"),
        n_positive=("n_fire_px", lambda s: (s > 0).sum()),
        n_negative=("n_fire_px", lambda s: (s == 0).sum()),
        total_fire_px=("n_fire_px", "sum"),
    )
    print(summary.to_string())

    # --- Сколько fire-пикселей в полных NaN ---
    full_nan = df[df["nan_group"] == "full_nan"]
    fire_in_full_nan = full_nan["n_fire_px"].sum()
    total_fire = df["n_fire_px"].sum()
    print(f"\n--- Потери ---")
    print(f"Пикселей горения в полных NaN-чипах: {fire_in_full_nan}")
    print(f"Всего пикселей горения: {total_fire}")
    print(f"Доля потерянных: {fire_in_full_nan / total_fire * 100:.2f}%")

    # --- Полные NaN-чипы с n_fire_px > 0 (это баг!) ---
    bad = full_nan[full_nan["n_fire_px"] > 0]
    if len(bad) > 0:
        print(f"\n⚠️  {len(bad)} ПОЛНОСТЬЮ NaN-ЧИПОВ ИМЕЮТ n_fire_px > 0:")
        print(bad[["chip_id", "n_fire_px", "valid_frac",
                   "cloud_frac", "landcover_top"]].to_string(index=False))
    else:
        print("\n✅ Все полностью NaN-чипы имеют n_fire_px = 0 (пустые негативы).")

    # --- Связь valid_frac и nan_group ---
    print(f"\n--- valid_frac по группам ---")
    print(df.groupby("nan_group")["valid_frac"].describe()[
        ["count", "mean", "min", "max"]
    ].to_string())


if __name__ == "__main__":
    meta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train/af/meta.csv")
    integ = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/af_integrity.csv")
    main(meta, integ)