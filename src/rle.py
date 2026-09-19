# RLE encode/decode для submission.csv

from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """
    Кодирует бинарную маску (H, W) в RLE-строку.
    Пиксели нумеруются построчно слева направо, сверху вниз, начиная с 1.
    Возвращает строку вида '20 2 34 3' (без кавычек) или '' если пикселей нет.
    """
    pixels = mask.flatten(order="C")
    # Добавляем нулевой барьер справа, чтобы серия корректно закрылась
    pixels = np.concatenate([[0], pixels, [0]])
    # Индексы переходов: где значение меняется
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    # runs теперь: [start1, len1, start2, len2, ...] в 1-индексации
    return " ".join(str(x) for x in runs)


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """
    Декодирует RLE-строку в бинарную маску (H, W).
    Пустая строка -> маска из нулей.
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if rle.strip():
        parts = rle.split()
        starts = np.asarray(parts[0::2], dtype=int) - 1  # в 0-индексацию
        lengths = np.asarray(parts[1::2], dtype=int)
        for s, l in zip(starts, lengths):
            mask[s : s + l] = 1
    return mask.reshape(shape, order="C")


def rle_encode_multiclass(mask: np.ndarray, class_id: int) -> str:
    """Кодирует только пиксели, где mask == class_id."""
    return rle_encode((mask == class_id).astype(np.uint8))


# ---------- Self-test на примере из постановки ----------
if __name__ == "__main__":
    # Из документа: маска 8x8, строка 3 столбцы 4-5 и строка 5 столбцы 2-4.
    # Ожидаемый RLE: "20 2 34 3"
    expected = "20 2 34 3"

    # Собираем маску вручную (1-индексация -> 0-индексация)
    # строка 3 (idx 2), столбцы 4-5 (idx 3,4): пиксели 20, 21
    # строка 5 (idx 4), столбцы 2-4 (idx 1,2,3): пиксели 34, 35, 36
    m = np.zeros((8, 8), dtype=np.uint8)
    m[2, 3] = 1
    m[2, 4] = 1
    m[4, 1] = 1
    m[4, 2] = 1
    m[4, 3] = 1

    got = rle_encode(m)
    print(f"encoded: '{got}'")
    print(f"expected: '{expected}'")
    assert got == expected, f"RLE encode FAIL: got '{got}'"
    print("✅ encode OK")

    decoded = rle_decode(got, (8, 8))
    assert np.array_equal(decoded, m), "RLE decode FAIL"
    print("✅ decode OK")

    # Проверка пустой маски
    assert rle_encode(np.zeros((8, 8), dtype=np.uint8)) == ""
    assert np.array_equal(rle_decode("", (8, 8)), np.zeros((8, 8), dtype=np.uint8))
    print("✅ empty mask OK")

    # Проверка мультикласса
    mc = np.zeros((4, 4), dtype=np.uint8)
    mc[0, 0] = 1
    mc[3, 3] = 3
    assert rle_encode_multiclass(mc, 1) == "1 1"
    assert rle_encode_multiclass(mc, 2) == ""
    assert rle_encode_multiclass(mc, 3) == "16 1"
    print("✅ multiclass OK")

    print("\n🎯 rle.py: все тесты пройдены")