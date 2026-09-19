"""
REST API для Fire Monitor.

Отдаёт те же данные, что Streamlit в режиме local, но через HTTP.
Контракт совпадает с RemoteDataSource из app.py.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="Fire Monitor API",
    version="1.0",
    description="Пространственно-временной запрос к данным AF и BS",
)

# CORS: разрешаем UI обращаться с любого origin (для локальной разработки)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Модели запроса/ответа
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    bbox: Optional[list[float]] = Field(
        default=None,
        description="[lon_min, lat_min, lon_max, lat_max] в WGS84",
    )
    polygon: Optional[dict] = Field(
        default=None,
        description="GeoJSON Polygon / MultiPolygon / Feature / FeatureCollection",
    )
    date_from: str = Field(..., description="YYYY-MM-DD")
    date_to: str = Field(..., description="YYYY-MM-DD")
    include: list[Literal["thermopoints", "burns", "summary"]] = Field(
        default=["thermopoints", "burns", "summary"],
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v):
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("bbox должен содержать 4 числа: [lon_min, lat_min, lon_max, lat_max]")
        lon_min, lat_min, lon_max, lat_max = v
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError("bbox: min должен быть меньше max")
        return v

    @field_validator("date_from", "date_to")
    @classmethod
    def _check_date(cls, v):
        try:
            date.fromisoformat(v[:10])
        except Exception:
            raise ValueError(f"Невалидная дата: {v}")
        return v


# ---------------------------------------------------------------------------
# Загрузка GeoJSON
# ---------------------------------------------------------------------------

def _read_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_manifest() -> dict:
    return _read_json("manifest.json")


def _load_thermopoints() -> dict:
    return _read_json("thermopoints.geojson") or {"type": "FeatureCollection", "features": []}


def _load_burns() -> dict:
    return _read_json("burns.geojson") or {"type": "FeatureCollection", "features": []}


# ---------------------------------------------------------------------------
# Фильтрация
# ---------------------------------------------------------------------------

def _parse_polygon(raw: dict) -> dict | None:
    if not raw:
        return None
    t = raw.get("type")
    if t == "Feature":
        return raw.get("geometry")
    if t == "FeatureCollection":
        features = raw.get("features") or []
        return features[0].get("geometry") if features else None
    if t in ("Polygon", "MultiPolygon"):
        return raw
    return None


def _in_bbox(lon: float, lat: float, bbox: list[float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _geom_in_area(geom: dict, bbox: list[float] | None, polygon: dict | None) -> bool:
    """Проверяет пересечение геометрии с областью запроса."""
    if not geom:
        return False

    gtype = geom.get("type")

    # Полигон: используем shapely
    if polygon:
        try:
            from shapely.geometry import shape
            return shape(geom).intersects(shape(polygon))
        except Exception:
            return False

    if not bbox:
        return True

    # Точка: простая проверка
    if gtype == "Point":
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            return False
        return _in_bbox(float(coords[0]), float(coords[1]), bbox)

    # Полигон/MultiPolygon: shapely
    if gtype in ("Polygon", "MultiPolygon"):
        try:
            from shapely.geometry import shape, box
            return shape(geom).intersects(
                box(bbox[0], bbox[1], bbox[2], bbox[3])
            )
        except Exception:
            return False

    return True


def _date_in_range(value: Any, date_from: str, date_to: str) -> bool:
    if not value:
        return True
    v = str(value)[:10]
    return date_from <= v <= date_to


def _burns_match_dates(props: dict, date_from: str, date_to: str) -> bool:
    d_pre = str(props.get("date_pre", ""))[:10]
    d_post = str(props.get("date_post", ""))[:10]
    if not d_pre and not d_post:
        return True
    lo = d_pre or d_post
    hi = d_post or d_pre
    return not (hi < date_from or lo > date_to)


def _filter_thermopoints(features: list[dict], req: QueryRequest) -> list[dict]:
    polygon = _parse_polygon(req.polygon) if req.polygon else None
    out = []
    for f in features:
        props = f.get("properties") or {}
        if not _date_in_range(props.get("date"), req.date_from, req.date_to):
            continue
        if not _geom_in_area(f.get("geometry") or {}, req.bbox, polygon):
            continue
        out.append(f)
    return out


def _filter_burns(features: list[dict], req: QueryRequest) -> list[dict]:
    polygon = _parse_polygon(req.polygon) if req.polygon else None
    out = []
    for f in features:
        props = f.get("properties") or {}
        if not _burns_match_dates(props, req.date_from, req.date_to):
            continue
        if not _geom_in_area(f.get("geometry") or {}, req.bbox, polygon):
            continue
        out.append(f)
    return out


def _build_summary(thermal: list[dict], burns: list[dict]) -> dict:
    total_area = 0.0
    by_sev = {
        "1": {"area_ha": 0.0, "count": 0, "label": "слабая"},
        "2": {"area_ha": 0.0, "count": 0, "label": "средняя"},
        "3": {"area_ha": 0.0, "count": 0, "label": "сильная"},
    }

    for f in burns:
        props = f.get("properties") or {}
        try:
            area = float(props.get("area_ha") or 0)
        except (TypeError, ValueError):
            area = 0.0
        total_area += area
        try:
            sev = int(props.get("severity"))
        except (TypeError, ValueError):
            continue
        key = str(sev)
        if key in by_sev:
            by_sev[key]["area_ha"] += area
            by_sev[key]["count"] += 1

    for v in by_sev.values():
        v["area_ha"] = round(v["area_ha"], 2)

    dates: list[str] = []
    for f in thermal:
        d = (f.get("properties") or {}).get("date")
        if d:
            dates.append(str(d)[:10])
    for f in burns:
        p = f.get("properties") or {}
        for k in ("date_pre", "date_post"):
            d = p.get(k)
            if d:
                dates.append(str(d)[:10])

    return {
        "n_thermopoints": len(thermal),
        "n_burn_polygons": len(burns),
        "total_burn_area_ha": round(total_area, 2),
        "by_severity": by_sev,
        "date_range_actual": {
            "min": min(dates) if dates else None,
            "max": max(dates) if dates else None,
        },
    }


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    m = _load_manifest()
    return {
        "status": "ok",
        "data_source": m.get("source", "unknown"),
        "date_range": m.get("date_range"),
    }


@app.get("/manifest")
def manifest() -> dict:
    return _load_manifest()


@app.post("/query")
def query(req: QueryRequest) -> JSONResponse:
    if not req.bbox and not req.polygon:
        raise HTTPException(
            status_code=400,
            detail="Нужно задать bbox или polygon",
        )

    thermal_all = _load_thermopoints().get("features") or []
    burns_all = _load_burns().get("features") or []

    thermal = _filter_thermopoints(thermal_all, req)
    burns = _filter_burns(burns_all, req)

    # Лимит: не более 10 000 фич в ответе
    total = len(thermal) + len(burns)
    if total > 10_000:
        raise HTTPException(
            status_code=413,
            detail=f"Слишком большой запрос ({total} фич). Сузьте bbox или интервал.",
        )

    include = set(req.include)
    response: dict[str, Any] = {
        "query": {
            "bbox": req.bbox,
            "date_from": req.date_from,
            "date_to": req.date_to,
        },
        "thermopoints": {"type": "FeatureCollection", "features": thermal} if "thermopoints" in include else None,
        "burns": {"type": "FeatureCollection", "features": burns} if "burns" in include else None,
        "summary": _build_summary(thermal, burns) if "summary" in include else None,
    }
    return JSONResponse(response)


def _export_params(
    bbox: Optional[str],
    polygon: Optional[str],
    date_from: str,
    date_to: str,
) -> QueryRequest:
    bbox_list = None
    if bbox:
        try:
            bbox_list = [float(x) for x in bbox.split(",")]
        except Exception:
            raise HTTPException(status_code=400, detail="bbox: ожидаются 4 числа через запятую")
        if len(bbox_list) != 4:
            raise HTTPException(status_code=400, detail="bbox: ожидаются 4 числа")

    polygon_dict = None
    if polygon:
        try:
            polygon_dict = json.loads(polygon)
        except Exception:
            raise HTTPException(status_code=400, detail="polygon: невалидный JSON")

    return QueryRequest(
        bbox=bbox_list,
        polygon=polygon_dict,
        date_from=date_from,
        date_to=date_to,
    )


@app.get("/export/thermopoints.geojson")
def export_thermopoints(
    bbox: Optional[str] = Query(default=None, description="lon_min,lat_min,lon_max,lat_max"),
    polygon: Optional[str] = Query(default=None, description="GeoJSON geometry"),
    date_from: str = Query(..., description="YYYY-MM-DD"),
    date_to: str = Query(..., description="YYYY-MM-DD"),
) -> Response:
    req = _export_params(bbox, polygon, date_from, date_to)
    thermal = _filter_thermopoints(_load_thermopoints().get("features") or [], req)
    body = json.dumps(
        {"type": "FeatureCollection", "features": thermal},
        ensure_ascii=False, indent=2,
    )
    return Response(
        content=body,
        media_type="application/geo+json",
        headers={"Content-Disposition": 'attachment; filename="thermopoints.geojson"'},
    )


@app.get("/export/burns.geojson")
def export_burns(
    bbox: Optional[str] = Query(default=None),
    polygon: Optional[str] = Query(default=None),
    date_from: str = Query(...),
    date_to: str = Query(...),
) -> Response:
    req = _export_params(bbox, polygon, date_from, date_to)
    burns = _filter_burns(_load_burns().get("features") or [], req)
    body = json.dumps(
        {"type": "FeatureCollection", "features": burns},
        ensure_ascii=False, indent=2,
    )
    return Response(
        content=body,
        media_type="application/geo+json",
        headers={"Content-Disposition": 'attachment; filename="burns.geojson"'},
    )


@app.get("/export/summary.json")
def export_summary(
    bbox: Optional[str] = Query(default=None),
    polygon: Optional[str] = Query(default=None),
    date_from: str = Query(...),
    date_to: str = Query(...),
) -> Response:
    req = _export_params(bbox, polygon, date_from, date_to)
    thermal = _filter_thermopoints(_load_thermopoints().get("features") or [], req)
    burns = _filter_burns(_load_burns().get("features") or [], req)
    summary = _build_summary(thermal, burns)
    return Response(
        content=json.dumps(summary, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="summary.json"'},
    )