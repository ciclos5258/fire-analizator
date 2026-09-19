"""Прогон BS-baseline на всех train-чипах: IoU_burn и mIoU_sev."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bs_baseline import predict_bs_chip  # noqa: E402


def load_gt(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        gt = src.read(1).astype(np.uint8)
        gt[gt == 255] = 0
    return gt


def main(data_dir: Path) -> None:
    meta = pd.read_csv(data_dir / "bs" / "meta.csv")
    print(f"BS-чипов в train: {len(meta)}")

    tp_b = fp_b = fn_b = 0
    tp_c = {1: 0, 2: 0, 3: 0}
    fp_c = {1: 0, 2: 0, 3: 0}
    fn_c = {1: 0, 2: 0, 3: 0}

    skipped = 0
    per_chip_iou = []

    for i, row in meta.iterrows():
        cid = row["chip_id"]
        s2_pre  = data_dir / "bs" / "sentinel2_pre"  / f"{cid}_Sentinel-2_pre.tif"
        s2_post = data_dir / "bs" / "sentinel2_post" / f"{cid}_Sentinel-2_post.tif"
        aux     = data_dir / "bs" / "aux"            / f"{cid}_AUX.tif"
        mask_p  = data_dir / "bs" / "masks"          / f"{cid}_MASK.tif"

        if not (s2_pre.exists() and s2_post.exists() and aux.exists() and mask_p.exists()):
            skipped += 1
            continue

        pred = predict_bs_chip(s2_pre, s2_post, aux)
        gt = load_gt(mask_p)

        # бинарный burn: класс >= 1
        p_b = pred >= 1
        g_b = gt >= 1
        tp_b += int((p_b & g_b).sum())
        fp_b += int((p_b & ~g_b).sum())
        fn_b += int((~p_b & g_b).sum())

        # по классам
        for c in (1, 2, 3):
            p = pred == c
            g = gt == c
            tp_c[c] += int((p & g).sum())
            fp_c[c] += int((p & ~g).sum())
            fn_c[c] += int((~p & g).sum())

        # IoU бинарный для чипа (для понимания распределения)
        inter = (p_b & g_b).sum()
        union = (p_b | g_b).sum()
        if union > 0:
            per_chip_iou.append(inter / union)

        if (i + 1) % 50 == 0:
            print(f"  обработано {i + 1}/{len(meta)}")

    print(f"\nПропущено: {skipped}")

    denom = tp_b + fp_b + fn_b
    iou_burn = tp_b / denom if denom > 0 else 1.0
    print(f"\n=== IoU_burn (микро) ===")
    print(f"TP={tp_b}, FP={fp_b}, FN={fn_b}")
    print(f"IoU_burn = {iou_burn:.4f}")

    print(f"\n=== mIoU_sev ===")
    ious = []
    for c in (1, 2, 3):
        d = tp_c[c] + fp_c[c] + fn_c[c]
        iou = tp_c[c] / d if d > 0 else 1.0
        ious.append(iou)
        print(f"  IoU({c}) = {iou:.4f}  (TP={tp_c[c]}, FP={fp_c[c]}, FN={fn_c[c]})")
    miou = float(np.mean(ious))
    print(f"mIoU_sev = {miou:.4f}")

    print(f"\n=== Вклад BS в Score ===")
    print(f"0.35 × {iou_burn:.4f} + 0.30 × {miou:.4f} = "
          f"{0.35 * iou_burn + 0.30 * miou:.4f}")

    if per_chip_iou:
        arr = np.array(per_chip_iou)
        print(f"\nPer-chip IoU_burn: median={np.median(arr):.3f}, "
              f"mean={arr.mean():.3f}, max={arr.max():.3f}")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train")
    main(data_dir)