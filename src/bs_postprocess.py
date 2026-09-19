"""Постобработка BS-маски: морфология + фильтр мелких компонент.

Убирает «соль-перец»: одиночные пиксели класса 1 на пашне, фенологию.
Гарь — это связное пятно, а не россыпь точек.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def postprocess_bs_mask(
    mask: np.ndarray,
    min_component_size: int = 25,
    opening_size: int = 3,
    closing_size: int = 3,
) -> np.ndarray:
    """
    Применяет морфологию и фильтр компонент к маске классов 0/1/2/3.

    Параметры:
        mask — uint8 (H, W), значения 0..3
        min_component_size — минимальный размер связной компоненты (пикселей)
        opening_size — размер ядра для бинарного открытия
        closing_size — размер ядра для бинарного закрытия

    Логика:
        1. Для каждого класса 1/2/3 отдельно:
           a. morphological opening — удаляет мелкий шум
           b. closing — заклеивает дырки внутри контура
           c. component filter — стирает компоненты < min_size
        2. Разрешает пересечения: приоритет у более сильного класса.
    """
    if not mask.any():
        return mask

    result = np.zeros_like(mask)

    # Порядок: сначала слабый, потом средний, потом сильный.
    # Если пиксель попал в 2 класса — побеждает более сильный.
    for cls in (1, 2, 3):
        binary = mask == cls

        if not binary.any():
            continue

        # 1. Открытие: убирает одиночные пиксели и тонкие линии
        if opening_size > 0:
            struct_open = np.ones((opening_size, opening_size), dtype=bool)
            binary = ndimage.binary_opening(binary, structure=struct_open)

        # 2. Закрытие: заклеивает мелкие дырки
        if closing_size > 0:
            struct_close = np.ones((closing_size, closing_size), dtype=bool)
            binary = ndimage.binary_closing(binary, structure=struct_close)

        # 3. Фильтр по размеру связных компонент
        if min_component_size > 0:
            labeled, n = ndimage.label(binary)
            if n > 0:
                sizes = ndimage.sum(binary, labeled, range(1, n + 1))
                keep = np.zeros(n + 1, dtype=bool)
                keep[1:] = sizes >= min_component_size
                binary = keep[labeled]

        # Записываем поверх (более сильный класс перезаписывает слабый)
        result[binary] = cls

    return result