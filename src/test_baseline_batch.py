# src/test_baseline_batch.py
import numpy as np
import rasterio
import pandas as pd
import sys, os

def baseline_af(I4, I5, valid):
    I4f = np.nan_to_num(I4, nan=0.0)
    I5f = np.nan_to_num(I5, nan=0.0)
    d45 = I4f - I5f
    return (I4f >= 330) & (d45 >= 15) & valid

def eval_chip(viirs_path, mask_path):
    with rasterio.open(viirs_path) as src:
        I4 = src.read(4); I5 = src.read(5)
    with rasterio.open(mask_path) as src:
        mask = src.read(1); mask[mask == 255] = 0
    valid = ~np.isnan(I4)
    pred = baseline_af(I4, I5, valid)
    
    tp = ((pred) & (mask == 1)).sum()
    fp = ((pred) & (mask == 0)).sum()
    fn = ((~pred) & (mask == 1)).sum()
    return tp, fp, fn

# Главный скрипт
meta = pd.read_csv(sys.argv[1])
data_dir = sys.argv[2]
results = []
for _, row in meta.iterrows():
    chip_id = row['chip_id']
    try:
        tp, fp, fn = eval_chip(
            f"{data_dir}/af/viirs/{chip_id}_VIIRS_I1-I5.tif",
            f"{data_dir}/af/masks/{chip_id}_MASK.tif"
        )
        results.append({
            'chip_id': chip_id,
            'n_fire_px': row['n_fire_px'],
            'tp': tp, 'fp': fp, 'fn': fn
        })
    except Exception as e:
        print(f"Ошибка на {chip_id}: {e}")

df = pd.DataFrame(results)
print(f"\n=== ИТОГО по {len(df)} чипам ===")
print(f"Суммарно: TP={df.tp.sum()}, FP={df.fp.sum()}, FN={df.fn.sum()}")

# Микро-усреднение
tp = df.tp.sum(); fp = df.fp.sum(); fn = df.fn.sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Топ-10 чипов по FP
print(f"\n=== Топ-10 чипов с FP ===")
print(df.nlargest(10, 'fp')[['chip_id', 'n_fire_px', 'tp', 'fp', 'fn']])

# Отрицательные чипы (n_fire_px=0)
neg = df[df.n_fire_px == 0]
print(f"\n=== Отрицательные чипы ({len(neg)}) ===")
print(f"Из них с FP > 0: {(neg.fp > 0).sum()}")
print(f"Суммарно FP на них: {neg.fp.sum()}")