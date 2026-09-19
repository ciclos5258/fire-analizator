"""MVP inference: AF по пороговому правилу, BS — пустой (заглушка)."""

from __future__ import annotations
from bs_baseline import predict_bs_chip

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rle import rle_encode  # noqa: E402


# ---------- AF-правило (grid search на train) ----------
AF_I4_MIN = 330.0
AF_D45_MIN = 25.0

# Индексы каналов в VIIRS-чипе (1-based для rasterio)
BAND_I4 = 4
BAND_I5 = 5

def find_bs_files(data_dir: Path, chip_id: str):
    """Ищет S2 pre, S2 post, aux."""
    candidates = [
        (data_dir / "bs" / "sentinel2_pre"  / f"{chip_id}_Sentinel-2_pre.tif",
         data_dir / "bs" / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif",
         data_dir / "bs" / "aux"            / f"{chip_id}_AUX.tif"),
        (data_dir / "bs" / "pre"  / f"{chip_id}_Sentinel-2_pre.tif",
         data_dir / "bs" / "post" / f"{chip_id}_Sentinel-2_post.tif",
         data_dir / "bs" / "aux"  / f"{chip_id}_AUX.tif"),
    ]
    for pre, post, aux in candidates:
        if pre.exists() and post.exists() and aux.exists():
            return pre, post, aux
    return None

def predict_af(viirs_path: Path) -> np.ndarray:
    """Возвращает бинарную маску (H, W) uint8 для AF-чипа.

    ВАЖНО: valid строится только по I4/I5. Каналы I1–I3 у части чипов
    полностью NaN (ночные пролёты), но это не мешает детекции горения.
    """
    with rasterio.open(viirs_path) as src:
        i4 = src.read(BAND_I4).astype(np.float32)
        i5 = src.read(BAND_I5).astype(np.float32)

    valid = ~np.isnan(i4) & ~np.isnan(i5)

    # Быстрый выход для полностью пустых чипов
    if not valid.any():
        return np.zeros(i4.shape, dtype=np.uint8)

    i4 = np.nan_to_num(i4, nan=0.0)
    i5 = np.nan_to_num(i5, nan=0.0)

    fire = valid & (i4 >= AF_I4_MIN) & ((i4 - i5) >= AF_D45_MIN)
    return fire.astype(np.uint8)


def load_sample(submission_csv: Path, data_dir: Path) -> pd.DataFrame:
    """Читает sample_submission.csv или собирает из meta.csv."""
    if submission_csv.exists():
        df = pd.read_csv(submission_csv)
        print(f"Шаблон найден: {submission_csv}")
        return df

    print(f"⚠️  {submission_csv} не найден, собираем из meta.csv")
    rows = []
    for kind in ("af", "bs"):
        meta_path = data_dir / kind / "meta.csv"
        if not meta_path.exists():
            print(f"   нет {meta_path}")
            continue
        df = pd.read_csv(meta_path)
        for cid in df["chip_id"]:
            if kind == "af":
                rows.append({"chip_id": cid, "class_id": 1, "rle": ""})
            else:
                for c in (1, 2, 3):
                    rows.append({"chip_id": cid, "class_id": c, "rle": ""})
    return pd.DataFrame(rows)


def find_viirs(data_dir: Path, chip_id: str) -> Path | None:
    """Ищет VIIRS-файл в нескольких возможных местах."""
    candidates = [
        data_dir / "af" / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif",
        data_dir / "af" / f"{chip_id}_VIIRS_I1-I5.tif",
        data_dir / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif",
        data_dir / f"{chip_id}_VIIRS_I1-I5.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    # Ищем sample_submission.csv
    sample_csv = data_dir / "sample_submission.csv"
    if not sample_csv.exists():
        sample_csv = data_dir.parent / "sample_submission.csv"

    df = load_sample(sample_csv, data_dir)
    print(f"Чипов в шаблоне: {len(df)}")

    # Уникальные chip_id для быстрого подсчёта
    unique_chips = df["chip_id"].unique().tolist()
    n_af = sum(1 for c in unique_chips if str(c).startswith("AF_"))
    n_bs = sum(1 for c in unique_chips if str(c).startswith("BS_"))
    print(f"  AF-чипов: {n_af}, BS-чипов: {n_bs}")

    af_cache: dict[str, str] = {}
    bs_cache: dict[str, np.ndarray] = {}
    n_af_processed = n_af_notfound = 0
    n_bs_processed = n_bs_notfound = n_bs_failed = 0

    rows_out = []
    n_af_processed = 0
    n_af_notfound = 0

    for _, row in df.iterrows():
        cid = str(row["chip_id"])
        cid_class = int(row["class_id"])

        if cid.startswith("AF_"):
            if cid_class != 1:
                # По контракту AF всегда class_id=1
                continue

            if cid in af_cache:
                rle = af_cache[cid]
            else:
                viirs = find_viirs(data_dir, cid)
                if viirs is None:
                    n_af_notfound += 1
                    rle = ""
                else:
                    mask = predict_af(viirs)
                    rle = rle_encode(mask)
                    n_af_processed += 1
                af_cache[cid] = rle

            rows_out.append({"chip_id": cid, "class_id": 1, "rle": rle})

        elif cid.startswith("BS_"):
            if cid not in bs_cache:
                files = find_bs_files(data_dir, cid)
                if files is None:
                    bs_cache[cid] = np.zeros((512, 512), dtype=np.uint8)
                    n_bs_notfound += 1
                else:
                    try:
                        bs_cache[cid] = predict_bs_chip(*files)
                        n_bs_processed += 1
                    except Exception as e:
                        print(f"  ⚠️  {cid}: {e}")
                        bs_cache[cid] = np.zeros((512, 512), dtype=np.uint8)
                        n_bs_failed += 1

            mask = bs_cache[cid]
            rle = rle_encode((mask == cid_class).astype(np.uint8))
            rows_out.append({"chip_id": cid, "class_id": cid_class, "rle": rle})

    # Запись в формате из постановки: кавычки только вокруг rle
    with open(output, "w", encoding="utf-8", newline="") as f:
        f.write("chip_id,class_id,rle\n")
        for r in rows_out:
            f.write(f'{r["chip_id"]},{int(r["class_id"])},"{r["rle"]}"\n')

    print(f"\nAF обработано:       {n_af_processed}")
    print(f"AF не найдено файла: {n_af_notfound}")
    print(f"BS обработано:       {n_bs_processed}")
    print(f"BS не найдено файла: {n_bs_notfound}")
    print(f"BS ошибок:           {n_bs_failed}")
    print(f"Строк всего:         {len(rows_out)} (ожидается {len(df)})")
    print(f"✅ Submission записан: {output}")

    if len(rows_out) != len(df):
        print("⚠️  Число строк не совпадает с шаблоном!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())