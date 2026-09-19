"""
Перебор порогов BS: меняем только weak-порог, medium/strong не трогаем.

Идея: FP >> TP в классе 1 — порог «слабой» слишком мягкий.
Поднимаем его и смотрим, что происходит с IoU_burn, mIoU_sev и Score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bs_baseline import (  # noqa: E402
    read_s2,
    read_landcover,
    compute_dnbr,
    EPS,
)


# Базовые пороги (medium/strong не трогаем)
BASE = {
    10: {"medium": 0.27, "strong": 0.66, "weak": 0.10},   # лес
    30: {"medium": 0.204, "strong": 0.386, "weak": 0.062},  # степь
    40: {"medium": 0.177, "strong": 0.38, "weak": 0.07},   # пашня
    90: {"medium": 0.331, "strong": 0.677, "weak": 0.075},  # пойма
}

# Варианты: сдвиг weak-порога на delta
WEAK_DELTAS = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10]


def classify_with_thresholds(dnbr, valid, landcover, thresholds):
    """thresholds: dict lc -> (weak, medium, strong)."""
    mask = np.zeros(dnbr.shape, dtype=np.uint8)
    for lc, (w, m, s) in thresholds.items():
        sel = (landcover == lc) & valid
        if not sel.any():
            continue
        d = dnbr[sel]
        cls = np.zeros(d.shape, dtype=np.uint8)
        cls[d >= w] = 1
        cls[d >= m] = 2
        cls[d >= s] = 3
        mask[sel] = cls
    return mask


def make_thresholds(delta: float) -> dict:
    """Сдвигаем weak на delta для всех landcover, medium/strong без изменений."""
    return {
        lc: (params["weak"] + delta, params["medium"], params["strong"])
        for lc, params in BASE.items()
    }


def main(data_dir: Path) -> None:
    meta = pd.read_csv(data_dir / "bs" / "meta.csv")
    print(f"BS-чипов в train: {len(meta)}")

    # --- Шаг 1: предзагрузка dNBR / valid / landcover ---
    # (тяжёлая часть — читаем TIF один раз)
    cache = []
    skipped = 0

    for i, row in meta.iterrows():
        cid = row["chip_id"]
        s2_pre  = data_dir / "bs" / "sentinel2_pre"  / f"{cid}_Sentinel-2_pre.tif"
        s2_post = data_dir / "bs" / "sentinel2_post" / f"{cid}_Sentinel-2_post.tif"
        aux     = data_dir / "bs" / "aux"            / f"{cid}_AUX.tif"
        mask_p  = data_dir / "bs" / "masks"          / f"{cid}_MASK.tif"

        if not (s2_pre.exists() and s2_post.exists() and aux.exists() and mask_p.exists()):
            skipped += 1
            continue

        pre = read_s2(s2_pre)
        post = read_s2(s2_post)
        dnbr, valid = compute_dnbr(pre, post)
        landcover = read_landcover(aux)

        with rasterio.open(mask_p) as src:
            gt = src.read(1).astype(np.uint8)
            gt[gt == 255] = 0

        cache.append((dnbr, valid, landcover, gt))

        if (i + 1) % 50 == 0:
            print(f"  загрузка {i + 1}/{len(meta)}")

    print(f"Загружено: {len(cache)}, пропущено: {skipped}\n")

    # --- Шаг 2: перебор вариантов ---
    print("=" * 78)
    print(f"{'Вариант':<14} {'IoU_burn':>9} {'mIoU':>8} "
          f"{'IoU(1)':>8} {'IoU(2)':>8} {'IoU(3)':>8} {'Score':>8}")
    print("=" * 78)

    results = []

    for delta in WEAK_DELTAS:
        thresholds = make_thresholds(delta)

        tp_b = fp_b = fn_b = 0
        tp_c = {1: 0, 2: 0, 3: 0}
        fp_c = {1: 0, 2: 0, 3: 0}
        fn_c = {1: 0, 2: 0, 3: 0}

        for dnbr, valid, landcover, gt in cache:
            pred = classify_with_thresholds(dnbr, valid, landcover, thresholds)

            p_b = pred >= 1
            g_b = gt >= 1
            tp_b += int((p_b & g_b).sum())
            fp_b += int((p_b & ~g_b).sum())
            fn_b += int((~p_b & g_b).sum())

            for c in (1, 2, 3):
                p = pred == c
                g = gt == c
                tp_c[c] += int((p & g).sum())
                fp_c[c] += int((p & ~g).sum())
                fn_c[c] += int((~p & g).sum())

        denom = tp_b + fp_b + fn_b
        iou_burn = tp_b / denom if denom > 0 else 1.0

        ious = []
        for c in (1, 2, 3):
            d = tp_c[c] + fp_c[c] + fn_c[c]
            ious.append(tp_c[c] / d if d > 0 else 1.0)
        miou = float(np.mean(ious))

        score = 0.35 * iou_burn + 0.30 * miou

        label = f"weak+{delta:.2f}"
        print(f"{label:<14} {iou_burn:>9.4f} {miou:>8.4f} "
              f"{ious[0]:>8.4f} {ious[1]:>8.4f} {ious[2]:>8.4f} {score:>8.4f}")

        results.append({
            "variant": label,
            "delta": delta,
            "iou_burn": iou_burn,
            "miou": miou,
            "iou_1": ious[0],
            "iou_2": ious[1],
            "iou_3": ious[2],
            "score": score,
            "tp_b": tp_b, "fp_b": fp_b, "fn_b": fn_b,
            "tp_1": tp_c[1], "fp_1": fp_c[1], "fn_1": fn_c[1],
        })

    # --- Шаг 3: выбор лучшего ---
    df = pd.DataFrame(results)
    best = df.loc[df["score"].idxmax()]

    print("\n" + "=" * 78)
    print("ИТОГ")
    print("=" * 78)
    print(f"Лучший вариант: {best['variant']}")
    print(f"  IoU_burn = {best['iou_burn']:.4f}  (baseline = {df.iloc[0]['iou_burn']:.4f})")
    print(f"  mIoU     = {best['miou']:.4f}  (baseline = {df.iloc[0]['miou']:.4f})")
    print(f"  Score    = {best['score']:.4f}  (baseline = {df.iloc[0]['score']:.4f})")
    print(f"  Прирост Score: +{best['score'] - df.iloc[0]['score']:.4f}")
    print(f"\n  IoU(1) = {best['iou_1']:.4f}")
    print(f"  IoU(2) = {best['iou_2']:.4f}")
    print(f"  IoU(3) = {best['iou_3']:.4f}")
    print(f"  TP_burn = {int(best['tp_b'])}, FP_burn = {int(best['fp_b'])}, "
          f"FN_burn = {int(best['fn_b'])}")

    # --- Сохранение ---
    out = Path("data/artifacts/tune_bs_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ Результаты сохранены: {out}")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train")
    main(data_dir)