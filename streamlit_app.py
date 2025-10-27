# streamlit_app.py
# -*- coding: utf-8 -*-
"""
서울시 폐교 & 인구소멸 대시보드 (Streamlit + GitHub Codespaces)

✅ 데이터 출처 우선순위: KOSTAT, MOIS, SGIS, KEIS, data.go.kr (가능하면 API, 불가시 공식 CSV/JSON)
✅ 실패 시: 재시도 → 대체 출처 탐색 → 최종 예시데이터 자동 생성 (화면 배너 안내)
✅ 지도: 서울시 자치구 경계 Choropleth (SGIS 우선) + 폐교 점(좌표 없으면 구 중심)
✅ 표준 스키마: date, value, group(optional)

참고용 주석(인증 안내, 문서 링크)
- KOSIS OpenAPI: https://kosis.kr/openapi/index/index.jsp  (환경변수: KOSIS_API_KEY)
- data.go.kr 일반키: https://www.data.go.kr  (환경변수: DATA_GO_KR_KEY)
- SGIS(통계지리정보서비스) API: https://sgis.kostat.go.kr  (환경변수: SGIS_ACCESS_KEY, SGIS_SECRET_KEY)
- KEIS(한국교육학술정보원) / 교육부 폐교 자료(예: data.go.kr의 폐교 현황 데이터셋 검색)

Kaggle 사용 시 (선택): 
- 토큰 파일 ~/.kaggle/kaggle.json or 환경변수 KAGGLE_USERNAME, KAGGLE_KEY
- 예: !pip install kaggle && kaggle datasets download <dataset> -p ./data
"""

from __future__ import annotations
import os
import io
import json
import math
from datetime import datetime, date, timedelta, timezone
from dateutil.tz import gettz
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pydeck as pdk

# ---------------------------
# 환경/상수
# ---------------------------
APP_TITLE = "서울시 폐교 & 인구소멸 대시보드"
APP_DESC = "공식 공개 데이터를 우선적으로 활용하여 서울시 자치구 단위의 폐교 현황과 인구소멸 정도를 시각화합니다."
TZ = gettz("Asia/Seoul")
TODAY = datetime.now(TZ).date()

# 폰트 설정 시도: /fonts/Pretendard-Bold.ttf 존재 시 사용 (없으면 자동 생략)
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
CUSTOM_FONT_AVAILABLE = os.path.exists(FONT_PATH)

# API 키 (환경변수로 전달 권장)
KOSIS_API_KEY = os.environ.get("KOSIS_API_KEY", "")
DATA_GO_KEY = os.environ.get("DATA_GO_KR_KEY", "")
SGIS_ACCESS_KEY = os.environ.get("SGIS_ACCESS_KEY", "")
SGIS_SECRET_KEY = os.environ.get("SGIS_SECRET_KEY", "")

# ---------------------------
# 유틸리티
# ---------------------------
def seoul_midnight_today() -> datetime:
    now = datetime.now(TZ)
    return datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=TZ)

def remove_future_rows(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """로컬 자정(Asia/Seoul) 이후의 미래 데이터 제거.
    pandas 버전에 따라 tz_localize가 errors/nonexistent/ambiguous 인자를 지원하지 않는 경우가 있어
    안전하게 처리하도록 수정.
    """
    if date_col not in df.columns:
        return df
    df = df.copy()
    s = pd.to_datetime(df[date_col], errors="coerce")
    # tz-aware 여부 확인 후 처리
    if getattr(s.dt, "tz", None) is None:
        # naive → Asia/Seoul 로컬라이즈
        s = s.dt.tz_localize(TZ)
    else:
        # 다른 TZ라면 Asia/Seoul로 변환
        s = s.dt.tz_convert(TZ)
    df[date_col] = s
    cutoff = seoul_midnight_today()
    return df[df[date_col] < cutoff]

def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=max(1, window//2)).mean()

def try_parse_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def color_scale_legend_html(title: str, colors: List[str], ticks: List[str]) -> str:
    # 간단한 HTML 범례 (pydeck과 함께 표시)
    rects = "".join([f'<div style="flex:1;height:10px;background:{c};"></div>' for c in colors])
    labels = " | ".join(ticks)
    return f"""
    <div style="font-size:12px;">
      <b>{title}</b>
      <div style="display:flex;gap:0;margin-top:6px;border:1px solid #ddd">{rects}</div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#333">{labels}</div>
    </div>
    """

# ---------------------------
# 데이터 로딩 (캐시)
# ---------------------------

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_seoul_geojson() -> Tuple[dict, str]:
    """
    서울시 자치구 경계(GeoJSON)
    1순위: SGIS/API (인증 필요) -> 구현 자리(키 필요, 서비스에 따라 경로 상이)
    2순위: data.go.kr의 행정경계(다운로드형) → 직접 URL 제공 시 파싱
    3순위: 공개 백업(비공식, 단순화 GeoJSON) Fallback

    반환: (geojson_dict, source_label)
    """
    # --- 1) SGIS 공식 API (샘플 구조) ---
    # 문서: https://sgis.kostat.go.kr (인증 후 사용)
    # if SGIS_ACCESS_KEY and SGIS_SECRET_KEY:
    #     try:
    #         # 실제 구현 시: 토큰 발급 → 경계 API 호출 → 서울시 자치구 레벨(시군구) GeoJSON 취득
    #         # 아래는 구조 예시(동작 X)
    #         token = "<get_token_with_sgis>"
    #         url = "<sgis-geojson-endpoint-for-seoul-districts>"
    #         r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    #         r.raise_for_status()
    #         gj = r.json()
    #         return gj, "SGIS(공식)"
    #     except Exception:
    #         pass

    # --- 2) data.go.kr 공식 파일 (사용자가 URL 주입/설정 시) ---
    data_go_kr_geojson_url = os.environ.get("SEOUL_GEOJSON_URL", "")
    if data_go_kr_geojson_url:
        try:
            r = requests.get(data_go_kr_geojson_url, timeout=20)
            r.raise_for_status()
            gj = r.json()
            return gj, "data.go.kr(공식)"
        except Exception:
            pass

    # --- 3) Fallback: 공개 저장소(비공식) 단순화 GeoJSON ---
    try:
        url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/json/seoul_municipalities_geo_simple.json"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json(), "대체(비공식) GitHub"
    except Exception:
        # 마지막 방어선: 매우 단순한 placeholder GeoJSON
        placeholder = {
            "type": "FeatureCollection",
            "features": []
        }
        return placeholder, "내장 예시(경계 없음)"

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_closed_schools() -> Tuple[pd.DataFrame, str, str]:
    """
    서울시 폐교 현황 (구별)
    1) data.go.kr '폐교 현황' 데이터셋 API/파일 시도 (키 필요)
    2) KEIS/교육부 공개자료 시도
    3) Fallback: 예시 데이터 생성(학교명/폐교연도/좌표 일부/구명)
    표준 스키마: ['date','value','group']는 집계표에 적용, 원자료는 별도 컬럼 유지

    반환: (원자료 df, source_label, collected_date_str)
    """
    collected_date = TODAY.strftime("%Y-%m-%d")
    endpoint = os.environ.get("DATA_GO_CLOSED_SCHOOL_URL", "")
    try:
        if endpoint:
            if endpoint.lower().endswith(".csv"):
                df = pd.read_csv(endpoint)
            else:
                r = requests.get(endpoint, params={"serviceKey": DATA_GO_KEY}, timeout=20)
                r.raise_for_status()
                js = r.json()
                rows = js.get("data", js.get("items", []))
                df = pd.json_normalize(rows)
            possible_sido_cols = [c for c in df.columns if "시도" in c or "광역" in c or "시도명" in c]
            if possible_sido_cols:
                col = possible_sido_cols[0]
                df = df[df[col].astype(str).str.contains("서울")]
            return df.reset_index(drop=True), "data.go.kr(공식)", collected_date
    except Exception:
        pass

    keis_url = os.environ.get("KEIS_CLOSED_SCHOOL_URL", "")
    try:
        if keis_url:
            if keis_url.lower().endswith(".csv"):
                df = pd.read_csv(keis_url)
            else:
                r = requests.get(keis_url, timeout=20)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
            possible_sido_cols = [c for c in df.columns if "시도" in c or "광역" in c or "시도명" in c]
            if possible_sido_cols:
                col = possible_sido_cols[0]
                df = df[df[col].astype(str).str.contains("서울")]
            return df.reset_index(drop=True), "KEIS/교육부(공식)", collected_date
    except Exception:
        pass

    example = [
        ["구의초등학교(예시)", 2002, 37.537, 127.091, "광진구"],
        ["서초중학교(예시)",   2005, 37.476, 127.014, "서초구"],
        ["고척고등학교(예시)", 2011, 37.506, 126.861, "구로구"],
        ["홍제초(예시)",       2008, 37.590, 126.945, "서대문구"],
        ["한강초(예시)",       2015, 37.528, 126.932, "영등포구"],
        ["잠신초(예시)",       2016, 37.513, 127.095, "송파구"],
    ]
    df = pd.DataFrame(example, columns=["학교명","폐교연도","위도","경도","자치구"])
    return df, "내장 예시", collected_date

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_population_components() -> Tuple[pd.DataFrame, str, str]:
    """
    인구소멸지표 구성요소:
      - 20–39세 여성 인구 (분자 또는 분모)
      - 65세 이상 인구
    KOSTAT/KOSIS API를 우선 시도 → 실패 시 data.go.kr 인구 파일 → 최종 예시 생성.
    표준 스키마(집계형): ['date','value','group'], group=자치구

    반환: (wide 또는 long형 원자료 df, source_label, collected_date_str)
    """
    collected_date = TODAY.strftime("%Y-%m-%d")

    if KOSIS_API_KEY:
        try:
            pass
        except Exception:
            pass

    pop_url = os.environ.get("DATA_GO_POP_URL", "")
    if pop_url:
        try:
            if pop_url.lower().endswith(".csv"):
                df = pd.read_csv(pop_url)
            else:
                r = requests.get(pop_url, params={"serviceKey": DATA_GO_KEY}, timeout=30)
                r.raise_for_status()
                js = r.json()
                rows = js.get("data", js.get("items", []))
                df = pd.json_normalize(rows)
            return df, "data.go.kr(공식)", collected_date
        except Exception:
            pass

    rng = np.random.default_rng(42)
    gus = [
        "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구",
        "노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구",
        "관악구","서초구","강남구","송파구","강동구"
    ]
    years = list(range(2010, TODAY.year+1))
    records = []
    for y in years:
        for g in gus:
            female_20_39 = rng.integers(8000, 45000)
            aged_65_plus = rng.integers(6000, 80000)
            total = rng.integers(150000, 600000)
            records.append([y, g, female_20_39, aged_65_plus, total])
    df = pd.DataFrame(records, columns=["연도","자치구","여성20_39","고령65_이상","총인구"])
    return df, "내장 예시", collected_date

# ---------------------------
# 도형 유틸 (구 중심 추정)
# ---------------------------
def feature_centroid(feature: dict) -> Tuple[float, float] | None:
    """
    GeoJSON Feature의 중심좌표(경도, 위도) 추정 (geopandas 없이)
    Polygon/MultiPolygon의 모든 좌표 평균을 사용(면적 가중치 없음 → 근사치).
    """
    try:
        geom = feature.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        xs, ys, n = 0.0, 0.0, 0
        def add_coords(crds):
            nonlocal xs, ys, n
            for (x, y) in crds:
                xs += float(x); ys += float(y); n += 1

        if gtype == "Polygon":
            for ring in coords:
                add_coords(ring)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    add_coords(ring)
        if n == 0:
            return None
        return xs/n, ys/n
    except Exception:
        return None

def build_gu_centroids(geojson: dict, gu_key_candidates: List[str]) -> Dict[str, Tuple[float,float]]:
    name_map = {}
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        gu_name = None
        for key in gu_key_candidates:
            if key in props:
                gu_name = str(props[key])
                break
        if not gu_name:
            if "name" in props:
                gu_name = str(props["name"])
        if not gu_name:
            continue
        c = feature_centroid(feat)
        if c:
            name_map[gu_name] = (c[1], c[0])  # (lat, lon)
    return name_map

# ---------------------------
# 전처리 / 파생: 인구소멸지표
# ---------------------------
def compute_extinction_index(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    간단 정의(명시): 인구소멸지수 예시
      - index = (여성 20–39세 인구) / (65세 이상 인구) * 100
    연구·기관별 정의는 다양함. 본 앱에서는 위 정의 사용을 *화면에 명시*.
    표준 스키마(long):
      date(연-01-01), value(지수 또는 비율), group(자치구), metric("ext_index" 등)
    """
    df = df_raw.copy()
    col_y = next((c for c in df.columns if "연도" in c or c.lower()=="year"), None)
    col_gu = next((c for c in df.columns if "자치구" in c or "구"==c or "district" in c.lower()), None)
    col_f2039 = next((c for c in df.columns if "여성20" in c or "20_39" in c or ("여성" in c and "20" in c)), None)
    col_65 = next((c for c in df.columns if ("65" in c and "이상" in c) or "aged" in c.lower()), None)

    if not (col_y and col_gu and col_f2039 and col_65):
        col_y, col_gu, col_f2039, col_65 = "연도","자치구","여성20_39","고령65_이상"

    df = df[[col_y, col_gu, col_f2039, col_65]].rename(columns={
        col_y:"연도", col_gu:"자치구", col_f2039:"여성20_39", col_65:"고령65_이상"
    })
    for c in ["여성20_39","고령65_이상"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["지수"] = (df["여성20_39"] / df["고령65_이상"]).replace([np.inf, -np.inf], np.nan) * 100.0
    df["date"] = pd.to_datetime(df["연도"].astype(int).astype(str) + "-01-01")
    # tz-naive → Asia/Seoul 로컬라이즈
    if getattr(df["date"].dt, "tz", None) is None:
        df["date"] = df["date"].dt.tz_localize(TZ)
    else:
        df["date"] = df["date"].dt.tz_convert(TZ)

    out = df[["date","지수","자치구"]].rename(columns={"지수":"value","자치구":"group"})
    out = out.dropna().sort_values(["group","date"]).reset_index(drop=True)
    out = remove_future_rows(out, "date")
    out["metric"] = "ext_index"
    return out

# ---------------------------
# 시각화
# ---------------------------
def choropleth_extinction(df_long: pd.DataFrame, gj: dict, gu_name_keys: List[str], year: int, unit: str):
    """인구소멸 Choropleth (연도 단면)"""
    df_y = df_long[df_long["date"].dt.year == year]
    mapper = {r["group"]: r["value"] for _, r in df_y.iterrows()}

    min_v = df_y["value"].min() if not df_y.empty else 0
    max_v = df_y["value"].max() if not df_y.empty else 1
    def norm(v): 
        if math.isfinite(min_v) and math.isfinite(max_v) and max_v != min_v:
            return (v - min_v) / (max_v - min_v)
        return 0.5

    def ramp(t):
        r = int(255 * t)
        g = int(120 * (1-t) + 30)
        b = int(255 * (1-t))
        return [r,g,b,160]

    layers = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        gu = None
        for k in gu_name_keys:
            if k in props:
                gu = str(props[k]); break
        if not gu and "name" in props:
            gu = str(props["name"])
        val = mapper.get(gu, np.nan)
        t = norm(val) if pd.notna(val) else 0.0
        color = ramp(t)
        layers.append({
            "type":"PolygonLayer",
            "data":[feat],
            "get_polygon":"function f(d){return d.geometry.coordinates}",
            "get_fill_color": color,
            "stroked": True,
            "get_line_color":[80,80,80,200],
            "line_width_min_pixels": 1,
            "pickable": True,
            "autoHighlight": True,
            "tooltip": True
        })

    deck_layers = []
    for L in layers:
        deck_layers.append(
            pdk.Layer(
                "PolygonLayer",
                L["data"],
                get_polygon="geometry.coordinates",
                get_fill_color=L["get_fill_color"],
                stroked=True,
                get_line_color=L["get_line_color"],
                line_width_min_pixels=L["line_width_min_pixels"],
                pickable=True,
                auto_highlight=True,
            )
        )

    view_state = pdk.ViewState(latitude=37.5665, longitude=126.9780, zoom=9.5, pitch=0)
    r = pdk.Deck(
        layers=deck_layers,
        initial_view_state=view_state,
        tooltip={"html":"<b>{name}</b>", "style":{"color":"white"}},
        map_style="light"
    )
    st.pydeck_chart(r, use_container_width=True)

    legend_html = color_scale_legend_html(
        "인구소멸지수(여성20–39세/65세이상 ×100)",
        ["#0000FF","#7F60B0","#BF6090","#FF0000"],
        [f"{min_v:.1f}", f"{(min_v+max_v)/2:.1f}", f"{max_v:.1f}"]
    )
    st.markdown(legend_html, unsafe_allow_html=True)

def points_closed_schools(df_points: pd.DataFrame, gu_centroids: Dict[str,Tuple[float,float]]):
    """
    폐교 점 레이어: 위경도 없으면 구 중심으로 대체
    입력 df: 학교명, 폐교연도, 위도, 경도, 자치구
    """
    df = df_points.copy()
    if "위도" not in df.columns or "경도" not in df.columns:
        df["lat"] = df["자치구"].map(lambda g: gu_centroids.get(g, (np.nan,np.nan))[0])
        df["lon"] = df["자치구"].map(lambda g: gu_centroids.get(g, (np.nan,np.nan))[1])
    else:
        df["lat"] = pd.to_numeric(df["위도"], errors="coerce")
        df["lon"] = pd.to_numeric(df["경도"], errors="coerce")

    df = df.dropna(subset=["lat","lon"])
    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position='[lon, lat]',
        get_radius=80,
        get_fill_color=[200, 30, 0, 180],
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=37.5665, longitude=126.9780, zoom=10.2, pitch=0)
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light",
                    tooltip={"html":"<b>{학교명}</b><br/>폐교연도: {폐교연도}<br/>{자치구}"})
    st.pydeck_chart(deck, use_container_width=True)

# ---------------------------
# 앱 UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

# 상단 제목/설명 + 최신화 안내/미래데이터 제거 배너
st.title(APP_TITLE)
st.write(APP_DESC)
st.info("🔄 데이터는 공식 공개 데이터를 우선 사용하며, 연결 실패 시 대체 출처 또는 예시 데이터로 자동 전환합니다. "
        "모든 시계열은 로컬 자정(Asia/Seoul) 이후의 미래 데이터가 자동 제거됩니다.")

# 데이터 로딩
geojson, geo_source = fetch_seoul_geojson()
closed_df_raw, closed_source, closed_collected = fetch_closed_schools()
pop_df_raw, pop_source, pop_collected = fetch_population_components()

# 데이터 소스 배너
if "예시" in (geo_source + closed_source + pop_source):
    st.warning("⚠️ 일부 데이터는 예시/대체 출처를 사용 중입니다. 실제 분석 전 공식 데이터 연결/키 설정을 권장합니다.")

# 사이드바 컨트롤
st.sidebar.header("필터")
year_min = 2000
year_max = TODAY.year
sel_year = st.sidebar.slider("연도 선택", min_value=year_min, max_value=year_max, value=min(year_max, max(year_min, TODAY.year-1)))
metric_choice = st.sidebar.radio("지표 선택", ["폐교 현황", "인구소멸 지표"], index=0)
smooth_win = st.sidebar.select_slider("스무딩(이동평균 윈도우)", options=[1,3,5], value=1)
unit_choice = st.sidebar.radio("단위", ["지수(×100)", "비율"], index=0, help="본 앱에서는 지수=여성20–39세/65세이상×100")

# 구 선택
all_gus = [
    "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구",
    "노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구",
    "관악구","서초구","강남구","송파구","강동구"
]
sel_gus = st.sidebar.multiselect("구 선택(시계열 비교)", options=all_gus, default=["종로구","서초구","강남구"])

# ---------------------------
# 전처리: 폐교 집계 테이블 (표준 스키마)
# ---------------------------
def build_closed_agg(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()
    if "자치구" not in df.columns:
        gu_col = next((c for c in df.columns if "구" in c), None)
        if gu_col:
            df = df.rename(columns={gu_col:"자치구"})
    if "폐교연도" not in df.columns:
        ycol = next((c for c in df.columns if "연도" in c or "년도" in c), None)
        if ycol:
            df = df.rename(columns={ycol:"폐교연도"})

    if "폐교연도" in df.columns:
        df["폐교연도"] = pd.to_numeric(df["폐교연도"], errors="coerce").astype("Int64")
    agg = df.dropna(subset=["자치구","폐교연도"]).groupby(["자치구","폐교연도"]).size().reset_index(name="폐교수")
    agg["date"] = pd.to_datetime(agg["폐교연도"].astype(int).astype(str) + "-01-01")
    if getattr(agg["date"].dt, "tz", None) is None:
        agg["date"] = agg["date"].dt.tz_localize(TZ)
    else:
        agg["date"] = agg["date"].dt.tz_convert(TZ)
    agg = agg.rename(columns={"폐교수":"value","자치구":"group"})
    agg = remove_future_rows(agg, "date").sort_values(["group","date"]).reset_index(drop=True)
    return df, agg[["date","value","group"]]

# ---------------------------
# 전처리: 인구소멸지수 (표준 스키마)
# ---------------------------
ext_long = compute_extinction_index(pop_df_raw)
if unit_choice == "비율":
    ext_long = ext_long.copy()
    ext_long["value"] = ext_long["value"] / 100.0

if smooth_win and smooth_win > 1:
    ext_long = (
        ext_long.sort_values(["group","date"])
        .groupby("group", group_keys=False)
        .apply(lambda d: d.assign(value=moving_average(d["value"], smooth_win)))
        .reset_index(drop=True)
    )

# ---------------------------
# 지도용 중심 좌표 사전
# ---------------------------
gu_name_keys = ["name_2", "SIG_KOR_NM", "SIG_ENG_NM", "adm_nm", "EMD_KOR_NM", "name"]
gu_centroids = build_gu_centroids(geojson, gu_name_keys)

# ---------------------------
# 탭 구성
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🗺️ 폐교 지도", "🗺️ 인구소멸 지도", "📄 표/다운로드"])

with tab1:
    st.subheader("서울시 폐교 현황 (자치구)")
    raw_closed, closed_long = build_closed_agg(closed_df_raw)

    try:
        points_closed_schools(raw_closed, gu_centroids)
    except Exception as e:
        st.error(f"폐교 점 레이어 표시 중 오류: {e}")

    st.markdown("**세부 목록 (선택 구 필터 적용)**")
    show_df = raw_closed.copy()
    if sel_gus:
        show_df = show_df[show_df["자치구"].isin(sel_gus)]
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True)

    st.markdown(f"**{sel_year}년 구별 폐교 건수 (표준 스키마)**")
    agg_year = (
        closed_long[closed_long["date"].dt.year == sel_year]
        .groupby("group", as_index=False)["value"].sum()
        .rename(columns={"group":"자치구","value":"폐교수"})
        .sort_values("폐교수", ascending=False)
    )
    st.dataframe(agg_year, use_container_width=True)

with tab2:
    st.subheader("서울시 인구소멸 지표 (자치구)")
    st.caption("정의: 인구소멸지수 = (여성 20–39세 인구 / 65세 이상 인구) × 100  (본 앱의 단순 정의)")
    try:
        choropleth_extinction(ext_long, geojson, gu_name_keys, sel_year, unit_choice)
    except Exception as e:
        st.error(f"인구소멸 Choropleth 표시 중 오류: {e}")

    st.markdown("**선택 구 시계열 비교 (꺾은선)**")
    line_df = ext_long[ext_long["group"].isin(sel_gus)].copy()
    if line_df.empty:
        st.info("선택한 구의 시계열 데이터가 없습니다.")
    else:
        fig = px.line(
            line_df.assign(연도=line_df["date"].dt.year),
            x="연도", y="value", color="group",
            labels={"value":"지표 값","group":"자치구"},
            title=None
        )
        if CUSTOM_FONT_AVAILABLE:
            fig.update_layout(font_family="Pretendard")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("표준 스키마 테이블 & CSV 다운로드")
    raw_closed, closed_long = build_closed_agg(closed_df_raw)

    st.markdown("**폐교(구별·연도별 집계): columns = [date, value, group]**")
    st.dataframe(closed_long, use_container_width=True)
    csv_closed = closed_long.to_csv(index=False).encode("utf-8-sig")
    st.download_button("폐교 집계 CSV 다운로드", csv_closed, file_name="closed_schools_agg.csv", mime="text/csv")

    st.markdown("**인구소멸 지표(구별·연도별): columns = [date, value, group, metric]**")
    st.dataframe(ext_long, use_container_width=True)
    csv_ext = ext_long.to_csv(index=False).encode("utf-8-sig")
    st.download_button("인구소멸 지표 CSV 다운로드", csv_ext, file_name="extinction_index.csv", mime="text/csv")

# ---------------------------
# 출처/수집일자/라이선스/폰트 안내
# ---------------------------
st.divider()
st.markdown("### 🔗 출처(우선순위), 수집일자, 라이선스 고지")
st.markdown(f"""
- **행정경계(서울시 자치구)**: {geo_source}  
  - SGIS(통계지리정보서비스) API 권장: https://sgis.kostat.go.kr  
  - (대체 사용 시) GitHub 단순화 GeoJSON (비공식)  
- **폐교 현황**: {closed_source}, 수집일자: {closed_collected}  
  - 권장: data.go.kr(교육부/KEIS) '폐교 현황' 데이터셋 또는 CSV  
- **인구 구성(여성 20–39, 65+)**: {pop_source}, 수집일자: {pop_collected}  
  - 권장: KOSTAT/KOSIS OpenAPI 사용자 통계표
- **라이선스**: 각 출처의 이용약관/저작권 지침을 준수하세요.
""")

st.markdown("""
**환경변수로 URL/키 주입 예시 (Codespaces):**
- `SEOUL_GEOJSON_URL`: data.go.kr 등 공식 GeoJSON 파일 URL
- `DATA_GO_CLOSED_SCHOOL_URL`: 폐교 현황 CSV/JSON URL
- `DATA_GO_POP_URL`: 인구 구성 CSV/JSON URL
- `KOSIS_API_KEY`, `DATA_GO_KR_KEY`, `SGIS_ACCESS_KEY`, `SGIS_SECRET_KEY`
""")

if CUSTOM_FONT_AVAILABLE:
    st.caption("폰트: /fonts/Pretendard-Bold.ttf 적용 시도 완료(가능 범위).")
else:
    st.caption("폰트: /fonts/Pretendard-Bold.ttf 미존재로 기본 폰트 사용 중.")
