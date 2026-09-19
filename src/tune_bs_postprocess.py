"""Перебор параметров постобработки BS. Ищем оптимальный вариант."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bs_baseline import predict_bs_chip  # noqa: E402
from bs_postprocess import postprocess_bs_mask  # noqa: E402


# (имя, opening, closing, min_size)
VARIANTS = [
    ("baseline",     0, 0,  0),   # без постобработки
    ("closing_only", 0, 3,  0),
    ("gentle_10",    0, 3, 10),
    ("medium_25",    0, 3, 25),
    ("gentle_no_close", 0, 0, 10),
    ("opening_only", 3, 0,  0),
]


def load_gt(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        gt = src.read(1).astype(np.uint8)
        gt[gt == 255] = 0
    return gt


def evaluate(preds, gts):
    tp_b = fp_b = fn_b = 0
    tp_c = {1: 0, 2: 0, 3: 0}
    fp_c = {1: 0, 2: 0, 3: 0}
    fn_c = {1: 0, 2: 0, 3: 0}
    for pred, gt in zip(preds, gts):
        p_b, g_b = pred >= 1, gt >= 1
        tp_b += int((p_b & g_b).sum())
        fp_b += int((p_b & ~g_b).sum())
        fn_b += int((~p_b & g_b).sum())
        for c in (1, 2, 3):
            p, g = pred == c, gt == c
            tp_c[c] += int((p & g).sum())
            fp_c[c] += int((p & ~g).sum())
            fn_c[c] += int((~p & g).sum())
    d = tp_b + fp_b + fn_b
    iou_b = tp_b / d if d else 1.0
    ious = []
    for c in (1, 2, 3):
        dd = tp_c[c] + fp_c[c] + fn_c[c]
        ious.append(tp_c[c] / dd if dd else 1.0)
    miou = float(np.mean(ious))
    score = 0.35 * iou_b + 0.30 * miou
    return dict(iou_burn=iou_b, miou=miou,
                iou_1=ious[0], iou_2=ious[1], iou_3=ious[2],
                score=score, fp_b=fp_b)


def main(data_dir: Path):
    meta = pd.read_csv(data_dir / "bs" / "meta.csv")
    print(f"BS-чипов: {len(meta)}")

    # --- Загрузка масок без постобработки (один раз) ---
    raw_masks, gts = [], []
    for i, row in meta.iterrows():
        cid = row["chip_id"]
        s2_pre  = data_dir / "bs" / "sentinel2_pre"  / f"{cid}_Sentinel-2_pre.tif"
        s2_post = data_dir / "bs" / "sentinel2_post" / f"{cid}_Sentinel-2_post.tif"
        aux     = data_dir / "bs" / "aux"            / f"{cid}_AUX.tif"
        mask_p  = data_dir / "bs" / "masks"          / f"{cid}_MASK.tif"
        if not all(p.exists() for p in (s2_pre, s2_post, aux, mask_p)):
            continue
        raw_masks.append(predict_bs_chip(s2_pre, s2_post, aux))
        gts.append(load_gt(mask_p))
        if (i + 1) % 50 == 0:
            print(f"  загрузка {i+1}/{len(meta)}")

    print()
    print("=" * 82)
    print(f"{'Вариант':<20} {'IoU_burn':>9} {'mIoU':>8} "
          f"{'IoU(1)':>8} {'IoU(2)':>8} {'IoU(3)':>8} {'Score':>8} {'FP_burn':>10}")
    print("=" * 82)

    results = []
    for name, opening, closing, min_size in VARIANTS:
        if name == "baseline":
            preds = raw_masks
        else:
            preds = [
                postprocess_bs_mask(
                    m,
                    opening_size=opening,
                    closing_size=closing,
                    min_component_size=min_size,
                )
                for m in raw_masks
            ]
        r = evaluate(preds, gts)
        print(f"{name:<20} {r['iou_burn']:>9.4f} {r['miou']:>8.4f} "
              f"{r['iou_1']:>8.4f} {r['iou_2']:>8.4f} {r['iou_3']:>8.4f} "
              f"{r['score']:>8.4f} {r['fp_b']:>10d}")
        results.append((name, r))

    # --- Лучший вариант ---
    best_name, best_r = max(results, key=lambda x: x[1]["score"])
    base_r = results[0][1]
    print()
    print(f"🏆 Лучший: {best_name}  Score={best_r['score']:.4f}  "
          f"(baseline={base_r['score']:.4f}, "
          f"прирост {best_r['score'] - base_r['score']:+.4f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train"))