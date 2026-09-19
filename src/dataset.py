import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class AFDataset(Dataset):
    """
    Dataset для Модуля 1 (Active Fire).
    Структура файлов (после распаковки):
        data/train/af/
        ├── viirs/{chip_id}_VIIRS_I1-I5.tif
        ├── masks/{chip_id}_MASK.tif
        ├── aux/{chip_id}_AUX.tif
        └── meta.csv
    """
    def __init__(self, meta_df, data_dir):
        self.meta = meta_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.viirs_dir = os.path.join(data_dir, 'viirs')
        self.masks_dir = os.path.join(data_dir, 'masks')
        self.aux_dir   = os.path.join(data_dir, 'aux')

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        chip_id = self.meta.iloc[idx]['chip_id']

        # --- Загрузка VIIRS ---
        viirs_path = os.path.join(self.viirs_dir, f'{chip_id}_VIIRS_I1-I5.tif')
        with rasterio.open(viirs_path) as src:
            viirs = src.read().astype(np.float32)   # (8, 256, 256)
        # Каналы 0-4: I1-I5, 5: solar_zenith, 6: sensor_zenith, 7: valid
        I1, I2, I3, I4, I5 = viirs[0], viirs[1], viirs[2], viirs[3], viirs[4]
        solar_zenith = viirs[5]
        sensor_zenith = viirs[6]

        # --- Загрузка AUX ---
        aux_path = os.path.join(self.aux_dir, f'{chip_id}_AUX.tif')
        with rasterio.open(aux_path) as src:
            aux = src.read().astype(np.float32)   # (5, 256, 256)
        landcover  = aux[0]
        dem        = aux[1]
        t2m        = aux[2]
        rh2m       = aux[3]
        wind_speed = aux[4]

        # --- Маска валидности (NaN в I1-I5) ---
        valid_mask = ~np.isnan(I1) & ~np.isnan(I2) & ~np.isnan(I3) \
                     & ~np.isnan(I4) & ~np.isnan(I5)
        valid_mask = valid_mask.astype(np.float32)

        # Заполняем NaN нулями (для подачи в модель)
        I1 = np.nan_to_num(I1, nan=0.0)
        I2 = np.nan_to_num(I2, nan=0.0)
        I3 = np.nan_to_num(I3, nan=0.0)
        I4 = np.nan_to_num(I4, nan=0.0)
        I5 = np.nan_to_num(I5, nan=0.0)

        # --- Сборка тензора признаков ---
        # 5 каналов VIIRS + solar_zenith + sensor_zenith + valid_mask
        # + 5 каналов AUX + 2 производных признака (I4-I5, I4/I5)
        features = np.stack([
            I1, I2, I3, I4, I5,
            solar_zenith, sensor_zenith, valid_mask,
            landcover, dem, t2m, rh2m, wind_speed,
            I4 - I5,                              # главный признак горения
            I4 / (I5 + 1e-6),                     # относительный контраст
        ], axis=0)                                # (15, 256, 256)

        # --- Загрузка маски ---
        mask_path = os.path.join(self.masks_dir, f'{chip_id}_MASK.tif')
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)   # (256, 256)
        # nodata=255 → заменяем на 0 (не пожар)
        mask[mask == 255] = 0

        return {
            'features': torch.from_numpy(features).float(),   # (15, 256, 256)
            'mask':     torch.from_numpy(mask).long(),         # (256, 256)
            'valid':    torch.from_numpy(valid_mask).float(),  # (256, 256)
            'chip_id':  chip_id,
        }