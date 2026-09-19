"""Проверка целостности AF-чипов: ищем битые строки/столбцы пикселей."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def analyze_chip(path: Path) -> dict:
    """Возвращает словарь с диагностикой одного чипа."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        i1 = src.read(4)
        n_bands = src.count

    nan_mask = np.isnan(i1)
    nan_total = int(nan_mask.sum())

    # Полностью NaN-строки и столбцы
    nan_rows = np.where(nan_mask.all(axis=1))[0].tolist()
    nan_cols = np.where(nan_mask.all(axis=0))[0].tolist()

    # Строки/столбцы, где NaN > 50% (частично побитые)
    partial_nan_rows = np.where((nan_mask.mean(axis=1) > 0.5))[0].tolist()
    partial_nan_cols = np.where((nan_mask.mean(axis=0) > 0.5))[0].tolist()

    # Углы: если NaN-область — клин, то верхние углы должны быть NaN
    corner_nan = {
        "tl": bool(nan_mask[0, 0]),
        "tr": bool(nan_mask[0, -1]),
        "bl": bool(nan_mask[-1, 0]),
        "br": bool(nan_mask[-1, -1]),
    }

    # Форма NaN-области: связная или нет? (грубая оценка)
    # Считаем долю NaN по каждой из 4 границ
    nan_top = float(nan_mask[0].mean())
    nan_bottom = float(nan_mask[-1].mean())
    nan_left = float(nan_mask[:, 0].mean())
    nan_right = float(nan_mask[:, -1].mean())

    return {
        "chip_id": path.stem.replace("_VIIRS_I1-I5", ""),
        "h": h,
        "w": w,
        "n_bands": n_bands,
        "nan_total": nan_total,
        "nan_frac": nan_total / (h * w),
        "n_nan_rows_full": len(nan_rows),
        "n_nan_cols_full": len(nan_cols),
        "n_partial_nan_rows": len(partial_nan_rows),
        "n_partial_nan_cols": len(partial_nan_cols),
        "corner_tl": corner_nan["tl"],
        "corner_tr": corner_nan["tr"],
        "corner_bl": corner_nan["bl"],
        "corner_br": corner_nan["br"],
        "nan_top": nan_top,
        "nan_bottom": nan_bottom,
        "nan_left": nan_left,
        "nan_right": nan_right,
    }


def main(viirs_dir: Path, output_csv: Path | None = None) -> None:
    chips = sorted(viirs_dir.glob("*_VIIRS_I1-I5.tif"))
    print(f"Найдено чипов: {len(chips)}\n")

    records = []
    for i, path in enumerate(chips, 1):
        try:
            rec = analyze_chip(path)
            records.append(rec)
        except Exception as e:
            print(f"❌ Ошибка чтения {path.name}: {e}")
        if i % 50 == 0:
            print(f"  обработано {i}/{len(chips)}")

    df = pd.DataFrame(records)

    # ---------- Сводная статистика ----------
    print("\n" + "=" * 70)
    print("СВОДНАЯ СТАТИСТИКА")
    print("=" * 70)

    print(f"\nВсего чипов: {len(df)}")
    print(f"Размеры: {df['h'].unique()} × {df['w'].unique()}")
    print(f"Каналов: {df['n_bands'].unique()}")

    print(f"\n--- NaN всего ---")
    print(f"Чипов с NaN: {(df['nan_total'] > 0).sum()} из {len(df)}")
    print(f"Чипов без NaN: {(df['nan_total'] == 0).sum()}")
    print(f"Медиана NaN: {df['nan_total'].median():.0f}")
    print(f"Мин/макс NaN: {df['nan_total'].min()} / {df['nan_total'].max()}")
    print(f"Уникальные значения nan_total (топ-10):")
    print(df["nan_total"].value_counts().head(10))

    print(f"\n--- Полностью NaN-строки/столбцы ---")
    print(f"Чипов с полными NaN-строками: {(df['n_nan_rows_full'] > 0).sum()}")
    print(f"Чипов с полными NaN-столбцами: {(df['n_nan_cols_full'] > 0).sum()}")
    if (df["n_nan_rows_full"] > 0).any():
        print(f"Макс полных NaN-строк в чипе: {df['n_nan_rows_full'].max()}")
    if (df["n_nan_cols_full"] > 0).any():
        print(f"Макс полных NaN-столбцов в чипе: {df['n_nan_cols_full'].max()}")

    print(f"\n--- Углы ---")
    for c in ["corner_tl", "corner_tr", "corner_bl", "corner_br"]:
        print(f"  {c}: NaN у {(df[c]).sum()} чипов")

    print(f"\n--- Границы (доля NaN по краю) ---")
    print(f"  top:    медиана {df['nan_top'].median():.3f}, макс {df['nan_top'].max():.3f}")
    print(f"  bottom: медиана {df['nan_bottom'].median():.3f}, макс {df['nan_bottom'].max():.3f}")
    print(f"  left:   медиана {df['nan_left'].median():.3f}, макс {df['nan_left'].max():.3f}")
    print(f"  right:  медиана {df['nan_right'].median():.3f}, макс {df['nan_right'].max():.3f}")

    # ---------- Подозрительные чипы ----------
    print(f"\n--- Подозрительные чипы (есть полные NaN-строки/столбцы) ---")
    suspicious = df[(df["n_nan_rows_full"] > 0) | (df["n_nan_cols_full"] > 0)]
    if len(suspicious) == 0:
        print("  Таких чипов нет — все NaN-области не вырождаются в целые линии.")
    else:
        print(suspicious[["chip_id", "nan_total", "n_nan_rows_full",
                          "n_nan_cols_full", "n_partial_nan_rows",
                          "n_partial_nan_cols"]].to_string(index=False))

    # ---------- Кластеризация по nan_total ----------
    print(f"\n--- Топ-5 чипов по числу NaN ---")
    top_nan = df.nlargest(5, "nan_total")[
        ["chip_id", "nan_total", "nan_frac", "n_nan_rows_full",
         "n_nan_cols_full", "n_partial_nan_rows", "n_partial_nan_cols"]
    ]
    print(top_nan.to_string(index=False))

    # ---------- Сохранение ----------
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Диагностика сохранена: {output_csv}")


if __name__ == "__main__":
    viirs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train/af/viirs")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/af_integrity.csv")
    main(viirs_dir, out)