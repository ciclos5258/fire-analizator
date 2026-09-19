import sys
import os
import numpy as np
import rasterio

def inspect(path):
    print(f"\n{'='*70}")
    print(f"ФАЙЛ: {os.path.basename(path)}")
    print(f"Размер: {os.path.getsize(path)/1024:.1f} KB")
    print(f"{'='*70}")
    
    with rasterio.open(path) as src:
        print(f"Каналов: {src.count}")
        print(f"Размер: {src.width} x {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Типы: {src.dtypes}")
        print(f"Nodata: {src.nodata}")
        print(f"Описание: {src.descriptions}")
        
        for i in range(src.count):
            b = src.read(i + 1)
            print(f"\n  Канал {i+1}:")
            if np.issubdtype(b.dtype, np.floating):
                nan_count = np.isnan(b).sum()
                if nan_count == b.size:
                    print(f"    ВСЁ NaN ({nan_count} пикселей)")
                    continue
                print(f"    min={np.nanmin(b):.2f}, max={np.nanmax(b):.2f}, mean={np.nanmean(b):.2f}")
                print(f"    NaN={nan_count}")
            else:
                print(f"    min={b.min()}, max={b.max()}, mean={b.mean():.2f}")
            uniq = np.unique(b)
            print(f"    уникальных: {len(uniq)}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        inspect(p)