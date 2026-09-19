"""BS baseline: dNBR + пороги по landcover + фильтр облаков через SCL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


# Индексы каналов Sentinel-2 (0-based)
S2_B8A = 6
S2_B12 = 8
S2_SCL = 9

SCL_INVALID = {3, 8, 9, 10}

# Пороги dNBR (weak, medium, strong) по ESA WorldCover
THRESHOLDS = {
    10: (0.13, 0.27, 0.66),      # лес
    30: (0.092, 0.204, 0.386),   # степь
    40: (0.10, 0.177, 0.38),     # пашня
    90: (0.105, 0.331, 0.677),   # пойма
}

S2_SCALE = 10000.0
EPS = 1e-6


def read_s2(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32) / S2_SCALE


def read_landcover(aux_path: Path) -> np.ndarray:
    with rasterio.open(aux_path) as src:
        aux = src.read()
    return aux[2].astype(np.int32)


def compute_dnbr(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b8a_pre = pre[S2_B8A]
    b12_pre = pre[S2_B12]
    b8a_post = post[S2_B8A]
    b12_post = post[S2_B12]

    nbr_pre = (b8a_pre - b12_pre) / (b8a_pre + b12_pre + EPS)
    nbr_post = (b8a_post - b12_post) / (b8a_post + b12_post + EPS)
    dnbr = nbr_pre - nbr_post

    scl_pre = pre[S2_SCL].astype(np.int32)
    scl_post = post[S2_SCL].astype(np.int32)
    invalid = np.isin(scl_pre, list(SCL_INVALID)) | np.isin(scl_post, list(SCL_INVALID))
    return dnbr, ~invalid


def classify_dnbr(dnbr: np.ndarray, valid: np.ndarray, landcover: np.ndarray) -> np.ndarray:
    mask = np.zeros(dnbr.shape, dtype=np.uint8)
    for lc, (w, m, s) in THRESHOLDS.items():
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


def predict_bs_chip(s2_pre: Path, s2_post: Path, aux: Path) -> np.ndarray:
    pre = read_s2(s2_pre)
    post = read_s2(s2_post)
    dnbr, valid = compute_dnbr(pre, post)
    landcover = read_landcover(aux)
    return classify_dnbr(dnbr, valid, landcover)


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bs_postprocess import postprocess_bs_mask

def predict_bs_chip_post(
    s2_pre: Path,
    s2_post: Path,
    aux: Path,
    use_postprocess: bool = True,
) -> np.ndarray:
    """Как predict_bs_chip, но с постобработкой."""
    mask = predict_bs_chip(s2_pre, s2_post, aux)
    if use_postprocess:
        mask = postprocess_bs_mask(mask)
    return mask