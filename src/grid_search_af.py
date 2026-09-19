# src/grid_search_af.py
"""
Grid search порогов AF-правила.

Считает F1 в двух режимах:
  - ALL: по всем 420 чипам (для сравнения с историей)
  - CLEAN: только по чипам с nan_frac < 0.99 (исключаем 127 битых)

В отчёт идёт CLEAN — честная оценка качества модели,
не искажённая отсутствующими данными.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# Порог, выше которого чип считается «битым» (полностью NaN или почти)
NAN_FRAC_BAD = 0.99


def load_chip(viirs_path: Path, mask_path: Path):
    with rasterio.open(viirs_path) as src:
        I1 = src.read(1)
        I3 = src.read(3)
        I4 = src.read(4)
        I5 = src.read(5)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask[mask == 255] = 0

    valid = ~np.isnan(I4)
    nan_frac = 1.0 - valid.mean()

    I3 = np.nan_to_num(I3, nan=0.0)
    I4 = np.nan_to_num(I4, nan=0.0)
    I5 = np.nan_to_num(I5, nan=0.0)
    return I3, I4, I5, valid, mask, float(nan_frac)


def evaluate(chips, i4_thr, d45_thr, i3_thr):
    """Считает TP/FP/FN и F1 по списку чипов."""
    tp = fp = fn = 0
    for I3, I4, I5, valid, mask, _ in chips:
        d45 = I4 - I5
        pred = (I4 >= i4_thr) & (d45 >= d45_thr) & (I3 <= i3_thr) & valid
        tp += int((pred & (mask == 1)).sum())
        fp += int((pred & (mask == 0)).sum())
        fn += int((~pred & (mask == 1)).sum())

    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f1


def grid_search(chips, label: str):
    """Перебирает пороги, печатает прогресс, возвращает лучшее."""
    best_f1 = 0.0
    best_params = None
    best_counts = None

    print(f"\n{'=' * 70}")
    print(f"GRID SEARCH: {label} (чипов: {len(chips)})")
    print(f"{'=' * 70}")

    for i4_thr in [325, 330, 335, 340, 345]:
        for d45_thr in [10, 15, 20, 25, 30]:
            for i3_thr in [0.3, 0.35, 0.4, 1.0]:
                tp, fp, fn, p, r, f1 = evaluate(chips, i4_thr, d45_thr, i3_thr)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (i4_thr, d45_thr, i3_thr)
                    best_counts = (tp, fp, fn, p, r)
                    print(f"  NEW BEST: I4>={i4_thr}, d45>={d45_thr}, "
                          f"I3<={i3_thr} → F1={f1:.4f}")

    tp, fp, fn, p, r = best_counts
    print(f"\n🏆 BEST [{label}]:")
    print(f"   I4>={best_params[0]}, d45>={best_params[1]}, I3<={best_params[2]}")
    print(f"   TP={tp}, FP={fp}, FN={fn}")
    print(f"   Precision={p:.4f}, Recall={r:.4f}, F1={best_f1:.4f}")
    return best_params, best_f1, (tp, fp, fn, p, r)


def main(meta_path: Path, data_dir: Path) -> None:
    meta = pd.read_csv(meta_path)
    print(f"Всего чипов в meta: {len(meta)}")

    # ---------- Загрузка всех чипов ----------
    chips_all = []   # (I3, I4, I5, valid, mask, nan_frac)
    skipped = []
    for _, row in meta.iterrows():
        chip_id = row["chip_id"]
        viirs = data_dir / "af" / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif"
        mask_p = data_dir / "af" / "masks" / f"{chip_id}_MASK.tif"
        try:
            I3, I4, I5, valid, mask, nan_frac = load_chip(viirs, mask_p)
            chips_all.append((I3, I4, I5, valid, mask, nan_frac))
        except Exception as e:
            skipped.append((chip_id, str(e)))

    if skipped:
        print(f"⚠️  Пропущено {len(skipped)} чипов (ошибки чтения):")
        for cid, err in skipped[:5]:
            print(f"   {cid}: {err}")

    # ---------- Разделение на чистые и битые ----------
    chips_clean = [c for c in chips_all if c[5] < NAN_FRAC_BAD]
    chips_bad   = [c for c in chips_all if c[5] >= NAN_FRAC_BAD]

    print(f"\nВсего загружено: {len(chips_all)}")
    print(f"  CLEAN (nan_frac < {NAN_FRAC_BAD}): {len(chips_clean)}")
    print(f"  BAD   (nan_frac ≥ {NAN_FRAC_BAD}): {len(chips_bad)}")

    # ---------- Сколько fire-пикселей теряем ----------
    fire_clean = sum(int((c[4] == 1).sum()) for c in chips_clean)
    fire_bad   = sum(int((c[4] == 1).sum()) for c in chips_bad)
    fire_all   = fire_clean + fire_bad
    print(f"\nFire-пикселей в CLEAN: {fire_clean}")
    print(f"Fire-пикселей в BAD:   {fire_bad}")
    print(f"Доля потерянных:       {fire_bad / fire_all * 100:.2f}%" if fire_all else "n/a")

    # ---------- Grid search на двух наборах ----------
    params_all,   f1_all,   counts_all   = grid_search(chips_all,   "ALL")
    params_clean, f1_clean, counts_clean = grid_search(chips_clean, "CLEAN")

    # ---------- Итоговое сравнение ----------
    print(f"\n{'=' * 70}")
    print("ИТОГ")
    print(f"{'=' * 70}")
    print(f"ALL:   F1={f1_all:.4f}  "
          f"TP={counts_all[0]}, FP={counts_all[1]}, FN={counts_all[2]}, "
          f"P={counts_all[3]:.4f}, R={counts_all[4]:.4f}")
    print(f"CLEAN: F1={f1_clean:.4f}  "
          f"TP={counts_clean[0]}, FP={counts_clean[1]}, FN={counts_clean[2]}, "
          f"P={counts_clean[3]:.4f}, R={counts_clean[4]:.4f}")

    if params_clean == params_all:
        print(f"\n✅ Лучшие пороги совпали: "
              f"I4>={params_clean[0]}, d45>={params_clean[1]}, I3<={params_clean[2]}")
    else:
        print(f"\n⚠️  Пороги различаются.")
        print(f"   ALL:   I4>={params_all[0]}, d45>={params_all[1]}, I3<={params_all[2]}")
        print(f"   CLEAN: I4>={params_clean[0]}, d45>={params_clean[1]}, I3<={params_clean[2]}")
        print(f"   Рекомендуется использовать CLEAN-пороги для инференса — "
              f"они не искажены битыми чипами.")

    # ---------- Сохранение результатов ----------
    out = Path("data/grid_search_af_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"set": "ALL",   "i4_min": params_all[0],   "d45_min": params_all[1],
         "i3_max": params_all[2],   "F1": f1_all,
         "TP": counts_all[0], "FP": counts_all[1], "FN": counts_all[2],
         "precision": counts_all[3], "recall": counts_all[4], "n_chips": len(chips_all)},
        {"set": "CLEAN", "i4_min": params_clean[0], "d45_min": params_clean[1],
         "i3_max": params_clean[2], "F1": f1_clean,
         "TP": counts_clean[0], "FP": counts_clean[1], "FN": counts_clean[2],
         "precision": counts_clean[3], "recall": counts_clean[4], "n_chips": len(chips_clean)},
    ]).to_csv(out, index=False)
    print(f"\n✅ Результаты сохранены: {out}")


if __name__ == "__main__":
    meta_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train/af/meta.csv")
    data_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/train")
    main(meta_path, data_dir)