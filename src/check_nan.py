import numpy as np
import rasterio
import sys

with rasterio.open(sys.argv[1]) as src:
    I1 = src.read(1)
    I2 = src.read(2)
    I3 = src.read(3)
    I4 = src.read(4)
    I5 = src.read(5)

nan_mask = np.isnan(I1)
print(f"NaN в I1: {nan_mask.sum()}")
print(f"NaN в I2: {np.isnan(I2).sum()}")
print(f"NaN в I3: {np.isnan(I3).sum()}")
print(f"NaN в I4: {np.isnan(I4).sum()}")
print(f"NaN в I5: {np.isnan(I5).sum()}")
print(f"\nСовпадают ли маски?")
print(f"I1==I2: {np.array_equal(np.isnan(I1), np.isnan(I2))}")
print(f"I1==I3: {np.array_equal(np.isnan(I1), np.isnan(I3))}")
print(f"I1==I4: {np.array_equal(np.isnan(I1), np.isnan(I4))}")
print(f"I1==I5: {np.array_equal(np.isnan(I1), np.isnan(I5))}")

# Визуально: где именно NaN?
print(f"\nРасположение NaN (первые 20 строк, 20 столбцов):")
print(nan_mask[:20, :20].astype(int))