from __future__ import annotations

import json
import math
import os
import random
import textwrap
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = 60

DATA_MODE = os.getenv("DATA_MODE", "remote").lower().strip()
if DATA_MODE not in ("local", "remote", "auto"):
    DATA_MODE = "auto"

DEMO_DATA_VERSION = 2

THEME = {
    "bg": "#0b1a2e",
    "bg_app": "#102643",
    "bg_panel": "#1a3a5c",
    "bg_sidebar": "#081524",
    "bg_input": "#0d2035",
    "accent": "#7588FD",
    "accent_hover": "#9aa8ff",
    "accent_deep": "#4a5ee0",
    "text": "#FFFFFF",
    "text_soft": "#b0c4de",
    "text_mute": "#7a8fa6",
    "border": "rgba(117, 136, 253, 0.30)",
    "border_soft": "rgba(117, 136, 253, 0.15)",
    "border_strong": "rgba(117, 136, 253, 0.55)",
    "thermopoint": "#ff4d4d",
    "blink_point": "#FFD700",
    "panel": "rgba(26, 58, 92, 0.55)",
    "shadow": "0 8px 32px rgba(0, 0, 0, 0.35)",
}

SEVERITY_META: dict[int, dict[str, str]] = {
    1: {
        "name": "Слабая",
        "fill": "rgba(255, 224, 138, 0.55)",
        "border": "#f59e0b",
        "emoji": "🟢",
    },
    2: {
        "name": "Средняя",
        "fill": "rgba(245, 158, 58, 0.60)",
        "border": "#ea580c",
        "emoji": "🟡",
    },
    3: {
        "name": "Сильная",
        "fill": "rgba(220, 38, 38, 0.65)",
        "border": "#991b1b",
        "emoji": "🔴",
    },
}

SEVERITY_LEGEND_COLORS = {
    1: "#ffe08a",
    2: "#f59e3a",
    3: "#dc2626",
}

SOURCE_LABELS = {
    "fake": "Demo data",
    "af_module": "AF module",
    "bs_module": "BS module",
    "combined": "Real data",
    "unknown": "Неизвестно",
}

FALLBACK_BBOX = [88.0, 54.0, 96.0, 58.0]
FALLBACK_DATE_MIN = date(2000, 1, 1)
FALLBACK_DATE_MAX = date(2030, 12, 31)

st.set_page_config(
    page_title="Fire Monitor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(markup: str) -> None:
    st.markdown(textwrap.dedent(markup), unsafe_allow_html=True)


def fmt_area(value: Any) -> str:
    try:
        return f"{float(value):,.2f} га".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def fmt_date(value: Any) -> str:
    if not value:
        return "—"
    try:
        return date.fromisoformat(str(value)[:10]).strftime("%d.%m.%Y")
    except Exception:
        return str(value)


def _zoom_from_bbox(bbox: list[float] | None) -> float:
    if not bbox or len(bbox) != 4:
        return 5.0

    lon_min, lat_min, lon_max, lat_max = bbox
    span_lon = abs(lon_max - lon_min)
    span_lat = abs(lat_max - lat_min)
    span = max(span_lon, span_lat, 1e-3)

    try:
        zoom = math.log2(360.0 / span) - 0.7
    except (ValueError, ZeroDivisionError):
        zoom = 5.0

    return max(1.0, min(16.0, zoom))


st.markdown(
    textwrap.dedent("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-app:        #0b1a2e;
            --bg-app-2:      #102643;
            --bg-panel:      #1a3a5c;
            --bg-sidebar:    #081524;
            --bg-input:      #0d2035;

            --border-soft:   rgba(117, 136, 253, 0.15);
            --border:        rgba(117, 136, 253, 0.30);
            --border-strong: rgba(117, 136, 253, 0.55);

            --accent:        #7588FD;
            --accent-hover:  #9aa8ff;
            --accent-deep:   #4a5ee0;

            --text:          #FFFFFF;
            --text-soft:     #b0c4de;
            --text-mute:     #7a8fa6;

            --shadow-blue:   0 8px 32px rgba(0, 0, 0, 0.35);
            --shadow-accent: 0 4px 16px rgba(117, 136, 253, 0.30);
        }

        html, body, .stApp, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', Roboto, sans-serif;
        }

        html, body, .stApp {
            background: var(--bg-app);
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 15% -10%,
                                rgba(117, 136, 253, 0.10), transparent 60%),
                radial-gradient(900px 500px at 100% 0%,
                                rgba(74, 94, 224, 0.08), transparent 55%),
                linear-gradient(180deg, #081524 0%, #102643 100%);
        }

        [data-testid="stHeader"] { background: transparent; }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp h5, .stApp h6 {
            color: #FFFFFF;
            font-weight: 700;
            letter-spacing: -0.3px;
            line-height: 1.25;
        }

        .stApp h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 0.75rem; }
        .stApp h2 {
            font-size: 1.4rem;
            color: var(--accent);
            border-bottom: 2px solid rgba(117, 136, 253, 0.30);
            padding-bottom: 8px;
            margin: 1.8rem 0 0.8rem;
        }
        .stApp h3 { font-size: 1.15rem; margin: 1.2rem 0 0.5rem; color: #FFFFFF; }
        .stApp h4 { font-size: 1rem; margin: 0.9rem 0 0.4rem; color: var(--text-soft); }

        .stApp h1 > a, .stApp h2 > a,
        .stApp h3 > a, .stApp h4 > a {
            display: none !important;
        }

        .stApp p, .stApp span, .stApp li, .stApp label, .stApp strong {
            color: var(--text-soft);
            line-height: 1.55;
        }

        .stApp a { color: var(--accent); }

        .stApp [data-testid="stCaptionContainer"], .stApp small {
            color: var(--text-mute) !important;
        }

        .stApp hr, [data-testid="stSidebar"] hr {
            border: none;
            border-top: 1px solid var(--border-soft);
            margin: 1rem 0;
        }

        .header { margin-bottom: 20px; }

        .header h1 {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.6px;
            margin: 0 0 6px;
            color: #FFFFFF;
        }

        .header h1::before {
            content: "";
            display: inline-block;
            width: 6px;
            height: 32px;
            background: var(--accent);
            border-radius: 3px;
            margin-right: 14px;
            vertical-align: -4px;
            box-shadow: 0 0 16px rgba(117, 136, 253, 0.7);
        }

        .header p {
            color: var(--text-mute);
            font-size: 1rem;
            margin: 0 0 0 20px;
        }

        .badge-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 6px 0 18px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid var(--border);
            background: rgba(117, 136, 253, 0.15);
            color: var(--accent);
            line-height: 1.4;
            white-space: nowrap;
        }

        .badge-ok {
            border-color: rgba(117, 136, 253, 0.55);
            background: rgba(117, 136, 253, 0.22);
            color: #FFFFFF;
        }

        .badge-warn {
            border-color: rgba(255, 200, 100, 0.50);
            background: rgba(255, 200, 100, 0.10);
            color: #FFC864;
        }

        .metric {
            background: rgba(26, 58, 92, 0.55);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 20px 24px;
            min-height: 120px;
            box-shadow: var(--shadow-blue);
            transition: all 0.25s ease;
        }

        .metric:hover {
            border-color: var(--border-strong);
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
        }

        .metric-title {
            color: var(--text-mute);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
        }

        .severity {
            background: rgba(26, 58, 92, 0.45);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 24px 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        .severity.sev-1 { border-left: 4px solid #ffe08a; }
        .severity.sev-2 { border-left: 4px solid #f59e3a; }
        .severity.sev-3 { border-left: 4px solid #dc2626; }

        .severity-name {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-soft);
        }

        .severity-area {
            font-size: 1.8rem;
            font-weight: 800;
            color: #FFFFFF;
        }

        .severity-percent {
            color: var(--text-mute);
            font-size: 0.8rem;
            margin-top: 4px;
        }

        .info-box {
            background: rgba(26, 58, 92, 0.40);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 20px 24px;
            color: var(--text-soft);
            box-shadow: var(--shadow-blue);
            line-height: 1.8;
        }

        .info-box b { color: var(--accent); }

        .info-box code {
            background: rgba(117, 136, 253, 0.15);
            color: var(--accent);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }

        .legend-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0;
            color: var(--text-soft);
            font-size: 0.9rem;
        }

        .legend-dot {
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            flex: 0 0 14px;
            border: 2px solid rgba(255, 255, 255, 0.35);
        }

        .legend-line {
            display: inline-block;
            width: 22px;
            height: 4px;
            border-radius: 2px;
            flex: 0 0 22px;
        }

        .footer-note {
            margin-top: 30px;
            padding-top: 14px;
            border-top: 1px solid var(--border-soft);
            color: var(--text-mute);
            font-size: 0.82rem;
            text-align: center;
            line-height: 1.6;
        }

        [data-testid="stSidebar"] {
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border-soft);
            color: var(--text-soft);
        }

        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #FFFFFF;
            border: none;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--text-mute) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
            color: var(--text-soft) !important;
        }

        [data-baseweb="input"],
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"],
        [data-baseweb="textarea"] {
            background-color: var(--bg-input) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }

        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        [data-baseweb="textarea"] textarea {
            color: var(--text) !important;
            background: transparent !important;
        }

        [data-baseweb="input"]:focus-within,
        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="textarea"]:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(117, 136, 253, 0.18) !important;
        }

        [data-baseweb="calendar"] {
            background-color: var(--bg-panel) !important;
            color: var(--text) !important;
        }

        .stButton button,
        .stDownloadButton button {
            background: var(--accent);
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.2rem;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-accent);
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: var(--accent-hover);
            box-shadow: 0 6px 22px rgba(117, 136, 253, 0.5);
            transform: translateY(-1px);
            color: #FFFFFF !important;
        }

        .stButton button:active,
        .stDownloadButton button:active {
            transform: translateY(0);
        }

        .stButton button:disabled,
        .stDownloadButton button:disabled {
            background: rgba(74, 94, 224, 0.30);
            color: rgba(255, 255, 255, 0.45) !important;
            box-shadow: none;
            cursor: not-allowed;
            transform: none;
        }

        .stAlert {
            background: rgba(117, 136, 253, 0.10) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px;
            color: var(--text) !important;
        }

        .stAlert [data-testid="stMarkdownContainer"] p,
        .stAlert [data-testid="stMarkdownContainer"] span {
            color: var(--text) !important;
        }

        [data-testid="stExpander"] {
            background: rgba(26, 58, 92, 0.40);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            color: var(--accent) !important;
            font-weight: 600;
        }

        [data-testid="stDataFrame"] {
            background: rgba(26, 58, 92, 0.40);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stPlotlyChart"] {
            background: rgba(11, 26, 46, 0.75);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 8px;
            box-shadow: var(--shadow-blue);
            overflow: hidden;
        }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--bg-app); }
        ::-webkit-scrollbar-thumb {
            background: rgba(117, 136, 253, 0.35);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(117, 136, 253, 0.6);
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """),
    unsafe_allow_html=True,
)


@dataclass
class FireQuery:
    bbox: list[float] | None
    polygon: dict | None
    date_from: date
    date_to: date


def _polygon_area_ha(coords: list[list[float]]) -> float:
    if not coords or len(coords) < 4:
        return 0.0

    pts = coords[:-1]
    if len(pts) < 3:
        return 0.0

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    r = 111_320.0
    cos_lat = math.cos(math.radians(cy))
    pts_m = [((p[0] - cx) * r * cos_lat, (p[1] - cy) * r) for p in pts]

    area_m2 = 0.0
    n = len(pts_m)
    for i in range(n):
        x1, y1 = pts_m[i]
        x2, y2 = pts_m[(i + 1) % n]
        area_m2 += x1 * y2 - x2 * y1

    return round(abs(area_m2) / 2 / 10_000, 2)


def generate_demo_data(data_dir: Path, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    thermal_file = data_dir / "thermopoints.geojson"
    burns_file = data_dir / "burns.geojson"
    manifest_file = data_dir / "manifest.json"

    need_regen = force

    if not need_regen:
        if not (
            thermal_file.exists() and burns_file.exists() and manifest_file.exists()
        ):
            need_regen = True
        else:
            try:
                m = json.loads(manifest_file.read_text(encoding="utf-8"))
                if int(m.get("demo_version", 0)) != DEMO_DATA_VERSION:
                    need_regen = True
            except Exception:
                need_regen = True

    if not need_regen:
        return

    from shapely.geometry import Point, Polygon

    rng = random.Random(42)

    bbox = [88.0, 54.0, 96.0, 58.0]
    date_min = date(2023, 4, 1)
    date_max = date(2023, 10, 31)
    region = "Красноярский край"
    satellites = ["Suomi NPP", "NOAA-20", "NOAA-21"]
    total_days = (date_max - date_min).days

    severity_labels = {1: "слабая", 2: "средняя", 3: "сильная"}
    severity_pool = [1] * 40 + [2] * 37 + [3] * 23

    burns: list[dict] = []
    n_burns = 150

    for i in range(n_burns):
        cx = rng.uniform(bbox[0] + 0.3, bbox[2] - 0.3)
        cy = rng.uniform(bbox[1] + 0.3, bbox[3] - 0.3)

        target_area_ha = rng.uniform(50, 2000)
        side_m = math.sqrt(target_area_ha) * 100
        side_lat_deg = side_m / 111_320.0
        side_lon_deg = side_m / (111_320.0 * math.cos(math.radians(cy)))

        n_vertices = rng.randint(6, 10)
        points: list[list[float]] = []
        for j in range(n_vertices):
            angle = 2 * math.pi * j / n_vertices
            radius_scale = rng.uniform(0.65, 1.0)
            px = cx + side_lon_deg * radius_scale * math.cos(angle)
            py = cy + side_lat_deg * radius_scale * math.sin(angle)
            points.append([round(px, 6), round(py, 6)])
        points.append(points[0])

        area_ha = _polygon_area_ha(points)
        if area_ha < 30:
            area_ha = round(rng.uniform(30, 80), 2)

        severity = rng.choice(severity_pool)

        day_offset = rng.randint(0, max(0, total_days - 25))
        date_pre = date_min + timedelta(days=day_offset)
        date_post = date_pre + timedelta(days=rng.randint(10, 30))

        chip_num = i // 3 + 1
        chip_id = f"BS_fake_{chip_num:06d}"

        burns.append(
            {
                "points": points,
                "area_ha": area_ha,
                "severity": severity,
                "date_pre": date_pre,
                "date_post": date_post,
                "chip_id": chip_id,
                "id": f"{chip_id}_sev{severity}",
            }
        )

    thermal: list[dict] = []

    def _make_thermal(lon, lat, d, chip_id, idx):
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        dt_iso = datetime(d.year, d.month, d.day, hour, minute).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 6), round(lat, 6)],
            },
            "properties": {
                "id": f"{chip_id}_{idx + 1:04d}",
                "chip_id": chip_id,
                "date": d.isoformat(),
                "datetime": dt_iso,
                "satellite": rng.choice(satellites),
                "confidence": round(rng.uniform(0.7, 1.0), 2),
                "n_fire_px": rng.randint(5, 120),
                "region": region,
            },
        }

    for burn in burns:
        poly = Polygon(burn["points"])
        minx, miny, maxx, maxy = poly.bounds
        span_days = max(1, (burn["date_post"] - burn["date_pre"]).days)

        n_inside = rng.randint(1, 3)
        for _ in range(n_inside):
            lon = lat = None
            for _ in range(40):
                px = rng.uniform(minx, maxx)
                py = rng.uniform(miny, maxy)
                if poly.contains(Point(px, py)):
                    lon, lat = px, py
                    break

            if lon is None:
                rp = poly.representative_point()
                lon, lat = rp.x, rp.y

            d = burn["date_pre"] + timedelta(days=rng.randint(0, span_days))
            chip_id = burn["chip_id"].replace("BS_", "AF_")
            thermal.append(_make_thermal(lon, lat, d, chip_id, len(thermal)))

    for _ in range(50):
        base = rng.random()
        offset_days = int((0.15 + 0.6 * base) * total_days) + rng.randint(-15, 15)
        offset_days = max(0, min(total_days, offset_days))
        d = date_min + timedelta(days=offset_days)

        lon = rng.uniform(bbox[0], bbox[2])
        lat = rng.uniform(bbox[1], bbox[3])

        chip_id = f"AF_fake_{rng.randint(1, 200):06d}"
        thermal.append(_make_thermal(lon, lat, d, chip_id, len(thermal)))

    thermal_file.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
                },
                "features": thermal,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    burn_features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [b["points"]]},
            "properties": {
                "id": b["id"],
                "chip_id": b["chip_id"],
                "severity": b["severity"],
                "severity_label": severity_labels[b["severity"]],
                "area_ha": b["area_ha"],
                "date_pre": b["date_pre"].isoformat(),
                "date_post": b["date_post"].isoformat(),
                "region": region,
            },
        }
        for b in burns
    ]

    burns_file.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
                },
                "features": burn_features,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest_file.write_text(
        json.dumps(
            {
                "version": "1.0",
                "demo_version": DEMO_DATA_VERSION,
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "fake",
                "date_range": {
                    "min": date_min.isoformat(),
                    "max": date_max.isoformat(),
                },
                "regions": [region],
                "counts": {
                    "thermopoints": len(thermal),
                    "burn_polygons": len(burn_features),
                },
                "bbox": bbox,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class BaseDataSource:
    name: str = "base"

    def get_manifest(self) -> dict:
        raise NotImplementedError

    def query(self, q: FireQuery) -> dict:
        raise NotImplementedError

    def export_thermopoints(self, q: FireQuery) -> bytes:
        raise NotImplementedError

    def export_burns(self, q: FireQuery) -> bytes:
        raise NotImplementedError

    def export_summary(self, q: FireQuery) -> bytes:
        raise NotImplementedError


class LocalDataSource(BaseDataSource):
    name = "Demo data"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def _read_json(self, filename: str) -> dict:
        path = self.data_dir / filename
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_manifest(self) -> dict:
        return self._read_json("manifest.json")

    def _in_query_area(self, geom: dict, q: FireQuery) -> bool:
        gtype = geom.get("type")

        if q.polygon:
            try:
                from shapely.geometry import shape

                poly = shape(q.polygon)
                return shape(geom).intersects(poly)
            except Exception:
                pass

        if q.bbox:
            lon_min, lat_min, lon_max, lat_max = q.bbox

            if gtype == "Point":
                coords = geom.get("coordinates") or []
                if len(coords) < 2:
                    return False
                lon, lat = coords[0], coords[1]
                return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

            if gtype in ("Polygon", "MultiPolygon"):
                try:
                    from shapely.geometry import shape, box

                    return shape(geom).intersects(
                        box(lon_min, lat_min, lon_max, lat_max)
                    )
                except Exception:
                    return False

        return True

    def _date_in_range(self, value, q: FireQuery) -> bool:
        if not value:
            return True
        v = str(value)[:10]
        return q.date_from.isoformat() <= v <= q.date_to.isoformat()

    def _burn_matches_dates(self, props: dict, q: FireQuery) -> bool:
        d_pre = str(props.get("date_pre", ""))[:10]
        d_post = str(props.get("date_post", ""))[:10]

        if not d_pre and not d_post:
            return True

        lo = d_pre or d_post
        hi = d_post or d_pre
        return not (hi < q.date_from.isoformat() or lo > q.date_to.isoformat())

    def _filter_thermopoints(self, features, q):
        result = []
        for f in features:
            props = f.get("properties") or {}
            if not self._date_in_range(props.get("date"), q):
                continue
            if not self._in_query_area(f.get("geometry") or {}, q):
                continue
            result.append(f)
        return result

    def _filter_burns(self, features, q):
        result = []
        for f in features:
            props = f.get("properties") or {}
            if not self._burn_matches_dates(props, q):
                continue
            if not self._in_query_area(f.get("geometry") or {}, q):
                continue
            result.append(f)
        return result

    @staticmethod
    def _build_summary(thermal, burns) -> dict:
        total_area = 0.0
        by_severity = {
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
            if key in by_severity:
                by_severity[key]["area_ha"] += area
                by_severity[key]["count"] += 1

        for entry in by_severity.values():
            entry["area_ha"] = round(entry["area_ha"], 2)

        all_dates = []
        for f in thermal:
            d = (f.get("properties") or {}).get("date")
            if d:
                all_dates.append(str(d)[:10])
        for f in burns:
            props = f.get("properties") or {}
            for key in ("date_pre", "date_post"):
                d = props.get(key)
                if d:
                    all_dates.append(str(d)[:10])

        return {
            "n_thermopoints": len(thermal),
            "n_burn_polygons": len(burns),
            "total_burn_area_ha": round(total_area, 2),
            "by_severity": by_severity,
            "date_range_actual": {
                "min": min(all_dates) if all_dates else None,
                "max": max(all_dates) if all_dates else None,
            },
        }

    def query(self, q: FireQuery) -> dict:
        thermal_fc = self._read_json("thermopoints.geojson")
        burns_fc = self._read_json("burns.geojson")

        thermal = self._filter_thermopoints(thermal_fc.get("features") or [], q)
        burns = self._filter_burns(burns_fc.get("features") or [], q)

        return {
            "query": {
                "bbox": q.bbox,
                "polygon": q.polygon,
                "date_from": q.date_from.isoformat(),
                "date_to": q.date_to.isoformat(),
            },
            "thermopoints": {"type": "FeatureCollection", "features": thermal},
            "burns": {"type": "FeatureCollection", "features": burns},
            "summary": self._build_summary(thermal, burns),
        }

    def export_thermopoints(self, q: FireQuery) -> bytes:
        fc = self._read_json("thermopoints.geojson")
        thermal = self._filter_thermopoints(fc.get("features") or [], q)
        return json.dumps(
            {"type": "FeatureCollection", "features": thermal},
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

    def export_burns(self, q: FireQuery) -> bytes:
        fc = self._read_json("burns.geojson")
        burns = self._filter_burns(fc.get("features") or [], q)
        return json.dumps(
            {"type": "FeatureCollection", "features": burns},
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

    def export_summary(self, q: FireQuery) -> bytes:
        return json.dumps(
            self.query(q)["summary"],
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")


class RemoteDataSource(BaseDataSource):
    name = "API backend"

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            r = requests.request(method, url, timeout=self.timeout, **kwargs)
        except requests.ConnectionError:
            raise ApiError(0, f"API недоступен по адресу {self.base_url}")
        except requests.Timeout:
            raise ApiError(0, f"API не ответил за {self.timeout:.0f} секунд")
        except requests.RequestException as exc:
            raise ApiError(0, f"Сетевая ошибка: {exc}")

        if r.status_code >= 400:
            try:
                payload = r.json()
                detail = str(payload.get("detail", r.text[:300]))
            except Exception:
                detail = r.text[:300]
            raise ApiError(r.status_code, detail)

        return r

    def _export_params(self, q: FireQuery) -> dict:
        params = {
            "date_from": q.date_from.isoformat(),
            "date_to": q.date_to.isoformat(),
        }
        if q.polygon:
            params["polygon"] = json.dumps(q.polygon)
        elif q.bbox:
            params["bbox"] = ",".join(str(float(v)) for v in q.bbox)
        return params

    def get_manifest(self) -> dict:
        r = self._request("GET", "/manifest")
        data = r.json()
        return data if isinstance(data, dict) else {}

    def query(self, q: FireQuery) -> dict:
        payload = {
            "bbox": q.bbox,
            "polygon": q.polygon,
            "date_from": q.date_from.isoformat(),
            "date_to": q.date_to.isoformat(),
            "include": ["thermopoints", "burns", "summary"],
        }
        r = self._request("POST", "/query", json=payload)
        data = r.json()
        return data if isinstance(data, dict) else {}

    def export_thermopoints(self, q: FireQuery) -> bytes:
        return self._request(
            "GET",
            "/export/thermopoints.geojson",
            params=self._export_params(q),
        ).content

    def export_burns(self, q: FireQuery) -> bytes:
        return self._request(
            "GET",
            "/export/burns.geojson",
            params=self._export_params(q),
        ).content

    def export_summary(self, q: FireQuery) -> bytes:
        return self._request(
            "GET",
            "/export/summary.json",
            params=self._export_params(q),
        ).content


def resolve_data_source() -> BaseDataSource:
    local = LocalDataSource(DATA_DIR)

    if DATA_MODE == "local":
        return local
    if DATA_MODE == "remote":
        return RemoteDataSource(API_URL)

    remote = RemoteDataSource(API_URL)
    try:
        remote.get_manifest()
        return remote
    except ApiError:
        return local


def _hover_html(props: dict, title: str) -> str:
    rows = []
    for key, value in props.items():
        if value is None or value == "":
            continue
        safe_key = str(key).replace("<", "&lt;").replace(">", "&gt;")
        safe_val = str(value).replace("<", "&lt;").replace(">", "&gt;")
        rows.append(f"<b>{safe_key}:</b> {safe_val}")

    safe_title = str(title).replace("<", "&lt;").replace(">", "&gt;")
    return f"<b>{safe_title}</b><br>" + "<br>".join(rows)


def _polygon_exterior_rings(
    geometry: dict,
) -> list[list[tuple[float, float]]]:
    if not geometry:
        return []

    t = geometry.get("type")
    coords = geometry.get("coordinates") or []
    rings: list[list[tuple[float, float]]] = []

    def _coerce(ring) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for p in ring:
            if len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        if len(pts) >= 3 and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts

    if t == "Polygon":
        if coords:
            r = _coerce(coords[0])
            if len(r) >= 4:
                rings.append(r)
    elif t == "MultiPolygon":
        for poly in coords:
            if poly:
                r = _coerce(poly[0])
                if len(r) >= 4:
                    rings.append(r)

    return rings


def _iter_all_coords(fc: dict):
    for feature in (fc or {}).get("features") or []:
        geom = feature.get("geometry") or {}
        t = geom.get("type")
        coords = geom.get("coordinates") or []

        if t == "Point" and len(coords) >= 2:
            yield float(coords[0]), float(coords[1])

        elif t == "Polygon":
            for ring in coords:
                for point in ring:
                    if len(point) >= 2:
                        yield float(point[0]), float(point[1])

        elif t == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for point in ring:
                        if len(point) >= 2:
                            yield float(point[0]), float(point[1])


def _center_and_zoom(
    thermopoints_fc: dict,
    burns_fc: dict,
    bbox: list[float] | None,
) -> tuple[float, float, float]:
    if bbox and len(bbox) == 4:
        lon_min, lat_min, lon_max, lat_max = bbox
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        return center_lat, center_lon, _zoom_from_bbox(bbox)

    lats: list[float] = []
    lons: list[float] = []

    for fc in (thermopoints_fc, burns_fc):
        for lon, lat in _iter_all_coords(fc):
            lons.append(lon)
            lats.append(lat)

    if not lats or not lons:
        return 56.0, 92.0, 5.0

    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2

    span_lat = max(lats) - min(lats)
    span_lon = max(lons) - min(lons)
    span = max(span_lat, span_lon, 1e-3)

    try:
        zoom = math.log2(360.0 / span) - 0.7
    except (ValueError, ZeroDivisionError):
        zoom = 5.0

    return center_lat, center_lon, max(1.0, min(16.0, zoom))


def _split_thermopoints_by_strong_burns(
    thermal_fc: dict,
    burns_fc: dict,
) -> tuple[list[dict], list[dict]]:
    from shapely.geometry import shape, Point

    strong_polys = []
    for f in (burns_fc or {}).get("features") or []:
        props = f.get("properties") or {}
        try:
            sev = int(props.get("severity") or 0)
        except (TypeError, ValueError):
            continue
        if sev != 3:
            continue
        try:
            strong_polys.append(shape(f.get("geometry")))
        except Exception:
            continue

    outside: list[dict] = []
    inside: list[dict] = []

    for f in (thermal_fc or {}).get("features") or []:
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue

        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue

        pt = Point(float(coords[0]), float(coords[1]))

        is_inside = any(
            poly.contains(pt) or poly.intersects(pt) for poly in strong_polys
        )

        (inside if is_inside else outside).append(f)

    return outside, inside


def build_plotly_map(
    thermopoints_fc: dict,
    burns_fc: dict,
    bbox: list[float] | None,
    show_thermopoints: bool,
    show_burns: bool,
    dark_map: bool,
) -> tuple[go.Figure, int | None]:
    fig = go.Figure()
    trace_count = 0
    blink_index: int | None = None

    if show_burns:
        features = (burns_fc or {}).get("features") or []

        for feature in features:
            props = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}

            try:
                sev = int(props.get("severity") or 0)
            except (TypeError, ValueError):
                sev = 0

            meta = SEVERITY_META.get(sev)
            if meta is None:
                fill_color = "rgba(148, 163, 184, 0.40)"
                border_color = "#64748b"
                line_width = 3.0
            else:
                fill_color = meta["fill"]
                border_color = meta["border"]
                line_width = 4.0 if sev == 3 else 3.0

            hover = _hover_html(
                {
                    "ID": props.get("id"),
                    "chip_id": props.get("chip_id"),
                    "Степень": f"{sev} — {props.get('severity_label', '')}",
                    "Площадь": fmt_area(props.get("area_ha")),
                    "Дата «до»": fmt_date(props.get("date_pre")),
                    "Дата «после»": fmt_date(props.get("date_post")),
                    "Регион": props.get("region"),
                },
                title=f"Гарь {props.get('id', '')}",
            )

            for ring in _polygon_exterior_rings(geometry):
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]

                fig.add_trace(
                    go.Scattermap(
                        lon=lons,
                        lat=lats,
                        mode="lines",
                        fill="toself",
                        fillcolor=fill_color,
                        line=dict(color=border_color, width=line_width),
                        name=f"Степень: {sev}",
                        legendgroup=f"severity-{sev}",
                        showlegend=False,
                        text=[hover] * len(lons),
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
                trace_count += 1

    if show_thermopoints:
        outside_features, inside_features = _split_thermopoints_by_strong_burns(
            thermopoints_fc,
            burns_fc,
        )

        def _pack(features):
            lons_, lats_, hovers_ = [], [], []
            for f in features:
                geom = f.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if len(coords) < 2:
                    continue
                props = f.get("properties") or {}
                lons_.append(float(coords[0]))
                lats_.append(float(coords[1]))
                hovers_.append(
                    _hover_html(
                        {
                            "ID": props.get("id"),
                            "chip_id": props.get("chip_id"),
                            "Дата": fmt_date(props.get("date")),
                            "Пролёт (UTC)": props.get("datetime"),
                            "Спутник": props.get("satellite"),
                            "Уверенность": props.get("confidence"),
                            "Пикселей горения": props.get("n_fire_px"),
                            "Регион": props.get("region"),
                        },
                        title=f"Термоточка {props.get('id', '')}",
                    )
                )
            return lons_, lats_, hovers_

        lons_o, lats_o, hovers_o = _pack(outside_features)
        if lons_o:
            fig.add_trace(
                go.Scattermap(
                    lon=lons_o,
                    lat=lats_o,
                    mode="markers",
                    name="Термоточки",
                    legendgroup="thermal-normal",
                    showlegend=False,
                    marker=dict(
                        size=12,
                        color=THEME["thermopoint"],
                        opacity=0.95,
                    ),
                    text=hovers_o,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            trace_count += 1

        lons_i, lats_i, hovers_i = _pack(inside_features)
        if lons_i:
            fig.add_trace(
                go.Scattermap(
                    lon=lons_i,
                    lat=lats_i,
                    mode="markers",
                    name="В сильной гари",
                    legendgroup="thermal-strong",
                    showlegend=False,
                    marker=dict(
                        size=22,
                        color=THEME["blink_point"],
                        opacity=1.0,
                    ),
                    text=hovers_i,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            trace_count += 1
            blink_index = trace_count

    if bbox and len(bbox) == 4:
        lon_min, lat_min, lon_max, lat_max = bbox
        fig.add_trace(
            go.Scattermap(
                lon=[lon_min, lon_max, lon_max, lon_min, lon_min],
                lat=[lat_min, lat_min, lat_max, lat_max, lat_min],
                mode="lines",
                line=dict(color=THEME["accent"], width=4),
                name="Область запроса",
                legendgroup="query-area",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        trace_count += 1

    center_lat, center_lon, zoom = _center_and_zoom(
        thermopoints_fc,
        burns_fc,
        bbox,
    )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0b1a2e",
        plot_bgcolor="#0b1a2e",
        showlegend=False,
        hoverlabel=dict(
            align="left",
            bgcolor="#081524",
            bordercolor=THEME["accent"],
            font=dict(color="#FFFFFF", size=12),
        ),
        dragmode="pan",
    )

    return fig, blink_index


def render_plotly_map(
    fig: go.Figure,
    blink_trace_index: int | None = None,
    height: int = 620,
) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        },
        key="fire_monitor_map",
    )


def _manifest_date_bounds(manifest: dict) -> tuple[date, date]:
    rng = manifest.get("date_range") or {}
    d_min = FALLBACK_DATE_MIN
    d_max = FALLBACK_DATE_MAX

    try:
        if rng.get("min"):
            d_min = date.fromisoformat(str(rng["min"])[:10])
    except Exception:
        pass
    try:
        if rng.get("max"):
            d_max = date.fromisoformat(str(rng["max"])[:10])
    except Exception:
        pass

    if d_max < d_min:
        d_max = d_min
    return d_min, d_max


def _manifest_default_bbox(manifest: dict) -> list[float]:
    bbox = manifest.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            return [float(v) for v in bbox]
        except (TypeError, ValueError):
            pass
    return list(FALLBACK_BBOX)


def _parse_polygon_input(text: str) -> dict | None:
    if not text or not text.strip():
        return None

    parsed = json.loads(text)
    t = parsed.get("type")

    if t == "Feature":
        geom = parsed.get("geometry")
        if not geom:
            raise ValueError("У Feature нет geometry")
        return geom

    if t == "FeatureCollection":
        features = parsed.get("features") or []
        if not features:
            raise ValueError("Пустая FeatureCollection")
        geom = features[0].get("geometry")
        if not geom:
            raise ValueError("У первого Feature нет geometry")
        return geom

    if t in ("Polygon", "MultiPolygon"):
        return parsed

    raise ValueError("Ожидается Polygon, MultiPolygon, Feature или FeatureCollection")


def render_sidebar(manifest: dict, source: BaseDataSource) -> dict:
    st.sidebar.markdown("# 🔎 Запрос")
    st.sidebar.caption("Параметры пространственно-временного поиска")
    st.sidebar.divider()

    st.sidebar.markdown("### 🗄 Источник данных")

    source_kind = str(manifest.get("source", "unknown"))
    source_label = SOURCE_LABELS.get(source_kind, source_kind)

    mode_badge = "badge-ok" if source.name != "Demo data" else "badge-warn"

    render_html(f"""
        <div style="margin: -6px 0 12px 0;">
            <span class="badge {mode_badge}">{source.name}</span>
        </div>
    """)

    st.sidebar.caption(f"Данные: **{source_label}**")
    if source.name == "API backend":
        st.sidebar.caption(f"API: `{API_URL}`")
    else:
        st.sidebar.caption(f"Каталог: `{DATA_DIR}`")

    date_min, date_max = _manifest_date_bounds(manifest)

    st.sidebar.divider()
    st.sidebar.markdown("### 📍 Территория")

    area_type = st.sidebar.radio(
        "Тип области",
        ["bbox", "полигон"],
        help="bbox — прямоугольник. Полигон — GeoJSON-геометрия.",
    )

    bbox: list[float] | None = None
    polygon: dict | None = None
    spatial_ok = False

    if area_type == "bbox":
        st.sidebar.caption("Ограничивающий прямоугольник (WGS84, lon/lat)")

        default_bbox = _manifest_default_bbox(manifest)

        c1, c2 = st.sidebar.columns(2)
        lon_min = c1.number_input(
            "lon_min",
            value=float(default_bbox[0]),
            step=0.1,
            format="%.4f",
            key="q_lon_min",
        )
        lat_min = c2.number_input(
            "lat_min",
            value=float(default_bbox[1]),
            step=0.1,
            format="%.4f",
            key="q_lat_min",
        )
        lon_max = c1.number_input(
            "lon_max",
            value=float(default_bbox[2]),
            step=0.1,
            format="%.4f",
            key="q_lon_max",
        )
        lat_max = c2.number_input(
            "lat_max",
            value=float(default_bbox[3]),
            step=0.1,
            format="%.4f",
            key="q_lat_max",
        )

        if lon_min >= lon_max or lat_min >= lat_max:
            st.sidebar.error("Некорректный bbox: min должен быть меньше max.")
        else:
            bbox = [lon_min, lat_min, lon_max, lat_max]
            spatial_ok = True

    else:
        st.sidebar.caption("GeoJSON: Polygon / MultiPolygon / Feature")

        poly_text = st.sidebar.text_area(
            "Геометрия",
            value="",
            height=150,
            key="q_polygon",
            placeholder=(
                '{"type":"Polygon","coordinates":'
                "[[[88,54],[96,54],[96,58],[88,58],[88,54]]]}"
            ),
        )

        if poly_text.strip():
            try:
                polygon = _parse_polygon_input(poly_text)
                if polygon is None:
                    raise ValueError("Пустая геометрия")
                spatial_ok = True
                st.sidebar.success("Геометрия распознана.")
            except Exception as exc:
                st.sidebar.error(f"Невалидный GeoJSON: {exc}")
        else:
            st.sidebar.caption("Вставьте геометрию или переключитесь на bbox.")

    st.sidebar.divider()
    st.sidebar.markdown("### 📅 Период")

    date_from = st.sidebar.date_input(
        "Дата начала",
        value=date_min,
        min_value=FALLBACK_DATE_MIN,
        max_value=FALLBACK_DATE_MAX,
        key="q_date_from",
    )
    date_to = st.sidebar.date_input(
        "Дата окончания",
        value=date_max,
        min_value=FALLBACK_DATE_MIN,
        max_value=FALLBACK_DATE_MAX,
        key="q_date_to",
    )

    temporal_ok = date_from <= date_to
    if not temporal_ok:
        st.sidebar.error("Дата начала позже даты окончания.")

    st.sidebar.divider()
    st.sidebar.markdown("### 🗺 Отображение")

    show_thermopoints = st.sidebar.checkbox(
        "🔥 Термоточки (AF)",
        value=True,
        key="q_show_thermal",
    )
    show_burns = st.sidebar.checkbox(
        "▱ Контуры гарей (BS)",
        value=True,
        key="q_show_burns",
    )

    st.sidebar.divider()

    execute = st.sidebar.button(
        "🔍 Запросить",
        type="primary",
        use_container_width=True,
        disabled=not (spatial_ok and temporal_ok),
        key="q_execute",
    )

    if not spatial_ok:
        st.sidebar.caption("Задайте корректную область запроса.")
    elif not temporal_ok:
        st.sidebar.caption("Исправьте период.")

    return {
        "bbox": bbox,
        "polygon": polygon,
        "date_from": date_from,
        "date_to": date_to,
        "show_thermopoints": show_thermopoints,
        "show_burns": show_burns,
        "execute": execute,
        "spatial_ok": spatial_ok,
        "temporal_ok": temporal_ok,
    }


def render_source_error(exc: ApiError) -> None:
    if exc.status_code == 0:
        st.error(f"🔌 {exc.detail}")
    elif exc.status_code == 400:
        st.warning(f"⚠️ Невалидный запрос: {exc.detail}")
    elif exc.status_code == 413:
        st.warning("⚠️ Слишком большой запрос — сузьте bbox или период.")
    elif exc.status_code == 422:
        st.warning(f"⚠️ Ошибка валидации: {exc.detail}")
    elif exc.status_code >= 500:
        st.error(f"💥 Ошибка сервера ({exc.status_code}): {exc.detail}")
    else:
        st.error(f"Ошибка {exc.status_code}: {exc.detail}")


def _severity_entry(by_severity: dict, level: int) -> dict:
    return by_severity.get(str(level)) or by_severity.get(level) or {}


def render_analytics(summary: dict) -> None:
    st.markdown("## 📊 Аналитическая справка")

    n_thermopoints = summary.get("n_thermopoints", 0)
    n_burn_polygons = summary.get("n_burn_polygons", 0)
    total_burn_area_ha = summary.get("total_burn_area_ha", 0.0)

    col1, col2, col3 = st.columns(3)

    with col1:
        render_html(f"""
            <div class="metric">
                <div class="metric-title">Всего термоточек</div>
                <div class="metric-value">{fmt_int(n_thermopoints)}</div>
            </div>
        """)

    with col2:
        render_html(f"""
            <div class="metric">
                <div class="metric-title">Контуров гарей</div>
                <div class="metric-value">{fmt_int(n_burn_polygons)}</div>
            </div>
        """)

    with col3:
        render_html(f"""
            <div class="metric">
                <div class="metric-title">Общая площадь гари</div>
                <div class="metric-value">{fmt_area(total_burn_area_ha)}</div>
            </div>
        """)

    render_html("<br>")

    st.markdown("### Распределение площади по степени поражения")

    by_severity = summary.get("by_severity") or {}

    total_sev_area = sum(
        float(_severity_entry(by_severity, lvl).get("area_ha", 0) or 0)
        for lvl in (1, 2, 3)
    )

    cols = st.columns(3)
    for col, level in zip(cols, (1, 2, 3)):
        data = _severity_entry(by_severity, level)
        area = float(data.get("area_ha", 0) or 0)
        count = int(data.get("count", 0) or 0)
        label = data.get("label") or SEVERITY_META[level]["name"]
        pct = (area / total_sev_area * 100) if total_sev_area > 0 else 0.0

        with col:
            render_html(f"""
                <div class="severity sev-{level}">
                    <div class="severity-name">
                        {SEVERITY_META[level]["emoji"]} {label}
                    </div>
                    <div class="severity-area">{fmt_area(area)}</div>
                    <div class="severity-percent">
                        {pct:.1f}% площади · {fmt_int(count)} контуров
                    </div>
                </div>
            """)

    st.markdown("### Таблица распределения")

    rows = []
    for level in (1, 2, 3):
        data = _severity_entry(by_severity, level)
        rows.append(
            {
                "Класс": level,
                "Степень": data.get("label") or SEVERITY_META[level]["name"],
                "Площадь, га": round(float(data.get("area_ha", 0) or 0), 2),
                "Контуров": int(data.get("count", 0) or 0),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    actual = summary.get("date_range_actual") or {}
    if actual.get("min") or actual.get("max"):
        st.caption(
            "Фактический период данных: "
            f"{fmt_date(actual.get('min'))} — {fmt_date(actual.get('max'))}"
        )


def render_export(source: BaseDataSource, q: FireQuery) -> None:
    st.markdown("## 📥 Выгрузка результатов")

    st.write(
        "Скачайте результаты запроса в машиночитаемом формате. "
        + (
            "Файлы формируются локально."
            if source.name == "Demo data"
            else "Файлы формируются backend-API на сервере."
        )
    )

    suffix = f"{q.date_from:%Y%m%d}_{q.date_to:%Y%m%d}"
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            st.download_button(
                "⬇ Термоточки (GeoJSON)",
                data=source.export_thermopoints(q),
                file_name=f"thermopoints_{suffix}.geojson",
                mime="application/geo+json",
                type="primary",
                use_container_width=True,
                key="dl_thermal",
            )
        except ApiError as exc:
            st.warning(f"Термоточки недоступны: {exc.detail}")

    with col2:
        try:
            st.download_button(
                "⬇ Контуры гарей (GeoJSON)",
                data=source.export_burns(q),
                file_name=f"burns_{suffix}.geojson",
                mime="application/geo+json",
                type="primary",
                use_container_width=True,
                key="dl_burns",
            )
        except ApiError as exc:
            st.warning(f"Контуры недоступны: {exc.detail}")

    with col3:
        try:
            st.download_button(
                "⬇ Справка (JSON)",
                data=source.export_summary(q),
                file_name=f"summary_{suffix}.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
                key="dl_summary",
            )
        except ApiError as exc:
            st.warning(f"Справка недоступна: {exc.detail}")


def main():

    generate_demo_data(DATA_DIR)

    render_html("""
        <div class="header">
            <h1>Fire Monitor</h1>
            <p>Информационно-аналитический сервис мониторинга природных пожаров</p>
        </div>
    """)

    source = resolve_data_source()

    manifest: dict = {}
    manifest_error: ApiError | None = None
    try:
        manifest = source.get_manifest()
    except ApiError as exc:
        manifest_error = exc

    if manifest_error is not None:
        render_source_error(manifest_error)

    controls = render_sidebar(manifest, source)

    source_kind = str(manifest.get("source", "unknown"))
    source_label = SOURCE_LABELS.get(source_kind, source_kind)

    badge_class = "badge-ok" if source.name != "Demo data" else "badge-warn"

    render_html(f"""
        <div class="badge-row">
            <span class="badge {badge_class}">{source.name}</span>
            <span class="badge">Модуль 1 — AF (VIIRS I1–I5)</span>
            <span class="badge">Модуль 2 — BS (Sentinel-2 + Sentinel-1)</span>
        </div>
    """)

    if "last_response" not in st.session_state:
        st.session_state.last_response = None
        st.session_state.last_query = None
        st.session_state.last_source = None

    if controls["execute"] and controls["spatial_ok"] and controls["temporal_ok"]:
        query = FireQuery(
            bbox=controls["bbox"],
            polygon=controls["polygon"],
            date_from=controls["date_from"],
            date_to=controls["date_to"],
        )

        spinner_text = (
            "Читаю локальные данные…"
            if source.name == "Demo data"
            else "Запрашиваю данные у backend…"
        )

        with st.spinner(spinner_text):
            try:
                response = source.query(query)
                st.session_state.last_response = response
                st.session_state.last_query = query
                st.session_state.last_source = source
            except ApiError as exc:
                st.session_state.last_response = None
                st.session_state.last_query = None
                st.session_state.last_source = None
                render_source_error(exc)

    response = st.session_state.last_response
    last_query = st.session_state.last_query
    last_source = st.session_state.last_source or source

    if response is None:
        if manifest_error is None:
            if source.name == "Demo data":
                hint = (
                    "Работает в демо-режиме на локальных данных. "
                    "Задайте область и период в боковой панели и нажмите "
                    "<code>🔍 Запросить</code>."
                )
            else:
                hint = (
                    "Backend доступен. Задайте область и период в боковой "
                    "панели и нажмите <code>🔍 Запросить</code>."
                )

            render_html(f"""
                <div class="info-box">
                    <b>Готов к работе</b><br>
                    {hint}
                </div>
            """)
        return

    st.markdown("## 🛰 Результат запроса")

    if last_query is not None:
        if last_query.polygon:
            area_desc = "Пользовательский полигон (GeoJSON)"
        elif last_query.bbox:
            b = last_query.bbox
            area_desc = f"bbox [{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, {b[3]:.4f}]"
        else:
            area_desc = "не задана"
    else:
        area_desc = "—"

    thermopoints_fc = response.get("thermopoints") or {
        "type": "FeatureCollection",
        "features": [],
    }
    burns_fc = response.get("burns") or {
        "type": "FeatureCollection",
        "features": [],
    }
    summary = response.get("summary") or {}

    n_thermal = len(thermopoints_fc.get("features") or [])
    n_burns = len(burns_fc.get("features") or [])

    date_from_str = last_query.date_from.strftime("%d.%m.%Y") if last_query else "—"
    date_to_str = last_query.date_to.strftime("%d.%m.%Y") if last_query else "—"

    render_html(f"""
        <div class="info-box">
            <b>Параметры запроса</b><br>
            📅 Период:
                <code>{date_from_str}</code>
                — <code>{date_to_str}</code><br>
            📍 Территория: {area_desc}<br>
            🗄 Источник: {last_source.name}<br>
            🔥 Термоточек: <b>{fmt_int(n_thermal)}</b>
            &nbsp;·&nbsp;
            ▱ Контуров гарей: <b>{fmt_int(n_burns)}</b>
        </div>
    """)

    render_html("<br>")

    if n_thermal == 0 and n_burns == 0:
        st.info(
            "В заданной области и интервале дат ничего не найдено. "
            "Попробуйте расширить период или увеличить bbox."
        )

    st.markdown("## 🗺 Карта мониторинга")

    map_fig, blink_index = build_plotly_map(
        thermopoints_fc=thermopoints_fc,
        burns_fc=burns_fc,
        bbox=last_query.bbox if last_query else None,
        show_thermopoints=controls["show_thermopoints"],
        show_burns=controls["show_burns"],
        dark_map=False,
    )

    render_plotly_map(
        map_fig,
        blink_trace_index=blink_index,
        height=620,
    )

    render_html(f"""
        <div class="info-box">
            <b>Легенда</b>
            <div style="margin-top:10px;">
                <div class="legend-row">
                    <span class="legend-dot"
                          style="background:{THEME['thermopoint']};"></span>
                    <span>Термоточка — активное горение</span>
                </div>
                <div class="legend-row">
                    <span class="legend-dot"
                          style="background:{THEME['blink_point']};"></span>
                    <span>Термоточка внутри контура сильной гари</span>
                </div>
                <div class="legend-row">
                    <span class="legend-line"
                          style="background:{SEVERITY_LEGEND_COLORS[1]};"></span>
                    <span>Слабая степень поражения</span>
                </div>
                <div class="legend-row">
                    <span class="legend-line"
                          style="background:{SEVERITY_LEGEND_COLORS[2]};"></span>
                    <span>Средняя степень поражения</span>
                </div>
                <div class="legend-row">
                    <span class="legend-line"
                          style="background:{SEVERITY_LEGEND_COLORS[3]};"></span>
                    <span>Сильная степень поражения</span>
                </div>
                <div class="legend-row">
                    <span class="legend-line"
                          style="background:{THEME['accent']};"></span>
                    <span>Граница области запроса (bbox)</span>
                </div>
            </div>
        </div>
    """)

    render_html("<br>")

    render_analytics(summary)

    render_html("<br>")

    if last_query is not None:
        render_export(last_source, last_query)

    render_html(f"""
        <div class="footer-note">
            Fire Monitor · прототип аналитического слоя
            регионального мониторинга пожаров
            <br>
            Источник данных: {source_label}
            &nbsp;·&nbsp;
            Режим: <code>DATA_MODE={DATA_MODE}</code>
            &nbsp;·&nbsp;
            Сформировано: {datetime.now():%d.%m.%Y %H:%M}
        </div>
    """)


if __name__ == "__main__":
    main()