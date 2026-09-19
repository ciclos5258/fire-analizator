import numpy as np
import rasterio
import sys

with rasterio.open(sys.argv[1]) as src:
    I1, I2, I3, I4, I5 = src.read(1), src.read(2), src.read(3), src.read(4), src.read(5)

with rasterio.open(sys.argv[2]) as src:
    mask = src.read(1)
    mask[mask == 255] = 0

valid = ~np.isnan(I1)

# Заполняем NaN нулями
I4f = np.nan_to_num(I4, nan=0.0)
I5f = np.nan_to_num(I5, nan=0.0)
d45 = I4f - I5f

# Применяем правило
pred = (I4f >= 330) & (d45 >= 15) & valid

# Считаем метрики
tp = ((pred == 1) & (mask == 1)).sum()
fp = ((pred == 1) & (mask == 0)).sum()
fn = ((pred == 0) & (mask == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Пикселей в эталоне: {(mask == 1).sum()}")
print(f"Пикселей в предсказании: {pred.sum()}")
print(f"TP={tp}, FP={fp}, FN={fn}")
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Диагностика: как выглядит распределение I4 и d45 в эталонных пожарах vs фоне
fire_I4  = I4f[mask == 1]
fire_d45 = d45[mask == 1]
bg_I4    = I4f[(mask == 0) & valid]
bg_d45   = d45[(mask == 0) & valid]

print(f"\nI4 в пожарах:  min={fire_I4.min():.1f}, median={np.median(fire_I4):.1f}, max={fire_I4.max():.1f}")
print(f"I4 в фоне:     min={bg_I4.min():.1f}, median={np.median(bg_I4):.1f}, max={bg_I4.max():.1f}")
print(f"d45 в пожарах: min={fire_d45.min():.1f}, median={np.median(fire_d45):.1f}, max={fire_d45.max():.1f}")
print(f"d45 в фоне:    min={bg_d45.min():.1f}, median={np.median(bg_d45):.1f}, max={bg_d45.max():.1f}")