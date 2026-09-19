# Мониторинг природных пожаров — КосмоХакатон 2026

Решение кейса «Двухэтапный мониторинг природных пожаров по данным VIIRS, Sentinel-2 и Sentinel-1».

Команда **Две калеки** · Красноярск, 18–20 сентября 2026.

## Результат

| Метрика               | Значение   |
|-----------------------|------------|
| **Public Score**      | **0.5969** |
| F1 (AF, train)        | 0.8902     |
| IoU_burn (BS, train)  | 0.3935     |
| mIoU_sev (BS, train)  | 0.4107     |

Формула: `Score = 0.35 × F1_af + 0.35 × IoU_burn + 0.30 × mIoU_sev`.

## Что решает

Два независимых модуля + сервис, объединённые общим файлом ответа:

- **AF (Active Fire)** — по VIIRS-чипу (I1–I5 + aux, 256×256 @ 375 м) находит пиксели природного горения, отделяет техногенные термоаномалии. Выход: бинарная маска.
- **BS (Burn Severity)** — по паре Sentinel-2 pre/post, Sentinel-1 pre/post, aux (512×512 @ 20 м) выделяет контур гари и классифицирует степень поражения: `0` — не горело, `1` — слабая, `2` — средняя, `3` — сильная.
- **Сервис** — Streamlit UI + FastAPI, карта термоточек и контуров гарей, аналитическая справка, выгрузка GeoJSON.

## Схема решения

```
VIIRS TIF ──► AF-правило ──► бинарная маска ─┐
                                              │
S2 pre/post ──► dNBR + landcover ──► маска ──┤──► submission.csv (RLE)
S1 pre/post ──► (задел на будущее) ──────────┤
SCL pre/post ──► маска облаков ──────────────┘
                                              │
                                              ▼
                                          лидерборд 
```

## Быстрый старт

### Установка

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/Mac

### Для запуска инференса
pip install -r requirements.txt

### Для запуска сервиса
pip install -r service/requirements.txt
```

### Запуск инференса

```bash
python src/inference.py --data-dir data/test --output data/submission.csv
```

Ожидаемая структура `data/test/`:
```
data/test/
├── af/viirs/{chip_id}_VIIRS_I1-I5.tif    # 180 чипов
├── bs/sentinel2_pre/{chip_id}_Sentinel-2_pre.tif   # 89
├── bs/sentinel2_post/
├── bs/aux/{chip_id}_AUX.tif
├── meta.csv
└── sample_submission.csv
```

Скрипт обрабатывает **все** чипы из `sample_submission.csv`, включая облачные, завершается с кодом 0 без интерактивного ввода.

### Валидация

```bash
python src/validate_submission.py data/submission.csv --sample data/test/sample_submission.csv
```

Ожидаемый результат: `✅ Submission валиден: 447 строк данных`.

## Структура проекта

```
project/
├── src/
│   ├── inference.py                 # главная точка входа: генерирует submission
│   ├── rle.py                       # RLE encode/decode + self-test
│   ├── validate_submission.py       # валидатор формата
│   ├── bs_baseline.py               # BS: dNBR + landcover + облака
│   ├── grid_search_af.py            # подбор порогов AF
│   ├── tune_bs_thresholds.py        # подбор порогов BS
│   ├── bs_postprocess.py            # морфология (эксперимент)
│   ├── test_bs_all.py               # оценка BS на train
│   ├── inspect_af.py, inspect_bs.py # ручная инспекция чипов
│   └── diagnose_af_bands.py         # диагностика NaN по каналам
├── service/                         # Streamlit UI + FastAPI
│   ├── app.py
│   ├── api.py
│   ├── data/                        # GeoJSON (генерируется или кладётся)
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── train/                       # обучающая выборка
│   ├── test/                        # тестовая выборка
│   ├── artifacts/                   # метрики, диагностика (CSV)
│   └── zipka/                       # оригиналы архивов
├── reports/
│   └── Отчёт по кейсу.md
├── requirements.txt
└── README.md
```

## Подход

### AF — пороговое правило

```python
valid = ~np.isnan(I4) & ~np.isnan(I5)
fire  = valid & (I4 >= 330) & (I4 - I5 >= 25)
```

**Физика:** очаг горения (600–1200 K) даёт контраст с фоном (~300 K) на порядки выше в канале I4 (3.74 мкм), чем в I5 (11.45 мкм). Разность `I4 − I5` — главный признак.

**Пороги подобраны grid search** на 420 обучающих чипах: перебор `I4 ∈ {325…345}`, `d45 ∈ {10…30}`, `I3 ∈ {0.3…1.0}`. Лучший F1 = 0.8902 при `I4 ≥ 330, d45 ≥ 25`.

**Важно:** у 30% обучающих чипов каналы I1–I3 полностью NaN (ночные пролёты VIIRS). `valid` строится только по I4/I5 — детекция работает корректно.

### BS — dNBR + landcover

```
NBR = (B8A − B12) / (B8A + B12)
dNBR = NBR_pre − NBR_post
```

Пороги зависят от типа земного покрова (ESA WorldCover):

| Landcover | Слабая | Средняя | Сильная |
|-----------|--------|---------|---------|
| Лес | 0.13 | 0.27 | 0.66 |
| Степь | 0.092 | 0.204 | 0.386 |
| Пашня | 0.10 | 0.177 | 0.38 |
| Пойма | 0.105 | 0.331 | 0.677 |

Пороги для «слабой» степени сдвинуты на +0.03 относительно исходных из постановки — это дало +0.006 к Score (см. `src/tune_bs_thresholds.py`).

**Маска облаков:** SCL классы `{3, 8, 9, 10}` (cloud shadow, cloud, cirrus) — невалидны.

## Сервис

Запуск (см. `service/README.md`):

```bash
# Терминал 1: API
uvicorn service.api:app --host 0.0.0.0 --port 8000

# Терминал 2: UI
streamlit run service/app.py
```

Открыть: **http://localhost:8501**

**Функционал:**
- Приём запроса: bbox или GeoJSON-полигон + интервал дат.
- Карта термоточек и контуров гарей по 3 классам.
- Аналитическая справка: площадь в га, распределение по 3 степеням.
- Выгрузка в GeoJSON и JSON.
- REST API: `/query`, `/export/thermopoints.geojson`, `/export/burns.geojson`, `/export/summary.json`.

**Режимы:** `DATA_MODE = local` (файлы) / `remote` (API) / `auto`.

Сервис работает на заранее подготовленных GeoJSON — спутниковые данные в реальном времени не качаются. Демонстрация на синтетических данных, реальный GeoJSON-контракт готов к подключению результатов моделей.

## Формат submission

```csv
chip_id,class_id,rle
AF_te_000001,1,"20 2 34 3"
BS_te_000001,1,"100 5 200 3"
BS_te_000001,2,""
BS_te_000001,3,""
```

- **447 строк** данных (180 AF × 1 + 89 BS × 3).
- RLE: пары `start length` через пробел, в двойных кавычках.
- Пиксели нумеруются построчно слева направо, сверху вниз, начиная с 1.
- Пустой класс — `""`.

## Воспроизводимость

- Все зависимости зафиксированы в `requirements.txt`.
- Seed = 42 (numpy, random).
- Повторный запуск даёт тот же результат (детерминированный baseline).
- Абсолютных путей с машины разработчика нет.

**Требования к окружению:**
- Python 3.11+
- Зависимости: `numpy`, `pandas`, `rasterio`, `scipy`

**Данные для воспроизведения:**
- `fire-train-renamed.tar` (~2.3 ГБ) → `data/train/`
- `fire-test-renamed.tar` (~900 МБ) → `data/test/`

## Ограничения и планы развития

**Реализовано:**
- Пороговое AF-правило (F1 = 0.89 на train).
- dNBR + landcover для BS (IoU_burn = 0.39 на train).
- Полный пайплайн submission.
- Сервис на GeoJSON.

**Не реализовано (в планах):**
- **AF:** LightGBM на пиксельных признаках (ожидаемо F1 → 0.93), фильтрация техногенки по landcover/OSM, контекстный анализ (скользящее окно).
- **BS:** SAR-признаки (ΔVV, ΔVH), RdNBR, U-Net для классификации по форме, кросс-валидация порогов.
- **Отклонённые эксперименты:** морфологическая постобработка BS — opening/closing ухудшают IoU(1) на 0.03 (см. отчёт, раздел 9.4).

**Ключевые ограничения:**
- AF: FN = 1552 (слабые пожары), FP = 464 (техногенка).
- BS: FP >> TP в классе 1 (фенология, пашня) — фундаментальное ограничение dNBR без учёта исходной биомассы.
- Пороги подобраны на всей train-выборке без отложенной валидации.

## Документация

- `/README.md` - основная документация
- `service/README.md` — документация интерфейса.

## Команда

**Две калеки** · КосмоХакатон 2026 · hackrus.experts

````