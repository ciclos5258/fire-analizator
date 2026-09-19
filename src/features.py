def _safe_dnbr(b8a_pre, b12_pre, b8a_post, b12_post, eps=1e-8):
    nbr_pre = (b8a_pre - b12_pre) / (b8a_pre + b12_pre + eps)
    nbr_post = (b8a_post - b12_post) / (b8a_post + b12_post + eps)
    dnbr = nbr_pre - nbr_post
    dnbr = np.nan_to_num(dnbr, nan=0.0, posinf=0.0, neginf=0.0)
    dnbr = np.clip(dnbr, -1.0, 1.0)
    return dnbr.astype(np.float32)

def load_sentinel2(path):
    """Читает многоканальный TIF, возвращает массив (C, H, W)."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read()  # (channels, H, W)

def load_sentinel1(path):
    """Аналогично для SAR."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read()

def load_aux(path):
    """AUX: landcover, dem, ..."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read()

def compute_nbr(b8a, b12, eps=1e-8):
    return (b8a - b12) / (b8a + b12 + eps)