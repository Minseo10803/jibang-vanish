
# streamlit_app.py
# -*- coding: utf-8 -*-
"""
📊 Streamlit + GitHub Codespaces 데이터 대시보드 (공식 공개 데이터 전용)

[공식 공개 데이터 대시보드]
- 통계청(KOSIS) OpenAPI를 통해 시도(행정구역) 단위 인구 지표를 수집해 '인구소멸지수' 계산
- 실패/미인증 시: 내부 예시 데이터로 자동 대체 + 알림 표시
- GeoJSON: 시도 경계 (GitHub - southkorea/southkorea-maps)
"""

import os
import json
import pytz
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, date

# =========================
# 한국어 폰트 (Pretendard) 시도
# =========================
def inject_pretendard_css():
    font_path = "/fonts/Pretendard-Bold.ttf"
    if os.path.exists(font_path):
        st.markdown(
            f"""
            <style>
            @font-face {{
                font-family: 'Pretendard';
                src: url('file://{font_path}') format('truetype');
                font-weight: 700;
                font-style: normal;
            }}
            html, body, [class*="css"]  {{
                font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo',
                             'Noto Sans KR', 'Noto Sans', 'Malgun Gothic', sans-serif !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

inject_pretendard_css()

st.set_page_config(page_title="대한민국 인구소멸지수 대시보드", layout="wide")

# =========================
# 오늘 이후 데이터 제거
# =========================
KST = pytz.timezone("Asia/Seoul")
TODAY = datetime.now(KST).date()
THIS_YEAR = TODAY.year

def remove_future_rows(df, date_col="date"):
    def to_date(x):
        try:
            if str(x).isdigit() and len(str(x)) == 4:
                return date(int(x), 12, 31)
            return pd.to_datetime(x).date()
        except Exception:
            return pd.NaT
    df[date_col] = df[date_col].apply(to_date)
    return df[df[date_col] <= TODAY]

# =========================
# 시도 목록 및 색상
# =========================
SIDO_ORDER = [
    "서울특별시","부산광역시","대구광역시","인천광역시","광주광역시","대전광역시","울산광역시",
    "세종특별자치시","경기도","강원특별자치도","충청북도","충청남도","전라북도","전라남도",
    "경상북도","경상남도","제주특별자치도"
]
COLOR_BINS = [0, 0.5, 0.8, 1.0, 1.2, 10]
COLOR_SCALE = ["#A8E6A3", "#F9E79F", "#F5CBA7", "#F1948A", "#8B0000"]

# =========================
# GeoJSON 로드
# =========================
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea-provinces-2018-geo.json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"GeoJSON 불러오기 실패: {e}"

# =========================
# 예시 데이터 (공개 API 실패 시)
# =========================
@st.cache_data
def example_public_dataset(seed=42, start_year=2015, end_year=THIS_YEAR):
    rng = np.random.default_rng(seed)
    years = list(range(start_year, end_year + 1))
    rows = []
    for y in years:
        for region in SIDO_ORDER:
            total = rng.integers(300_000, 13_000_000)
            young_f = max(1, int(total * rng.uniform(0.09, 0.16)))
            old_65 = max(1, int(total * rng.uniform(0.12, 0.25)))
            rows.append({
                "year": y, "region": region, "total_pop": total,
                "young_female": young_f, "old_65plus": old_65
            })
    df = pd.DataFrame(rows)
    df["extinction_index"] = df["young_female"] / df["old_65plus"]
    df["date"] = df["year"].astype(str)
    df = remove_future_rows(df, "date")
    return df

# =========================
# KOSIS API (옵션)
# =========================
@st.cache_data
def fetch_kosis_population(api_key, start_year=2010, end_year=THIS_YEAR):
    try:
        url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
        params = {
            "method": "getList",
            "apiKey": api_key,
            "itmId": "T20F_39F",
            "objL1": "시도",
            "orgId": "101",
            "tblId": "DT_1B040A3",
            "prdSe": "Y",
            "startPrdDe": str(start_year),
            "endPrdDe": str(end_year),
            "format": "json"
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        return df
    except Exception:
        return None

# =========================
# 사이드바
# =========================
st.sidebar.header("옵션")
year_min = st.sidebar.number_input("시작 연도", min_value=1990, max_value=THIS_YEAR, value=2015)
year_max = st.sidebar.number_input("종료 연도", min_value=year_min, max_value=THIS_YEAR, value=THIS_YEAR)
smooth_window = st.sidebar.select_slider("스무딩(이동평균)", [1, 2, 3, 4, 5], value=1)
unit_scale = st.sidebar.selectbox("단위 변환(총인구)", ["명", "천 명", "만 명"], index=0)

def scale_values(series, unit):
    if unit == "천 명":
        return series / 1_000
    if unit == "만 명":
        return series / 10_000
    return series

# =========================
# 공개 데이터 대시보드
# =========================
st.title("🇰🇷 대한민국 인구소멸지수 대시보드")
st.caption("공식 공개 데이터(KOSIS) 기반, 실패 시 예시 데이터로 대체됩니다.")

api_key = os.getenv("KOSIS_API_KEY", "")
if api_key:
    df = fetch_kosis_population(api_key, start_year=year_min, end_year=year_max)
else:
    df = None

if df is None:
    df = example_public_dataset(start_year=year_min, end_year=year_max)
    st.info("KOSIS API 인증키가 없거나 연결 실패로 예시 데이터를 사용합니다.")

gj, gj_err = load_geojson()
if gj_err:
    st.warning(gj_err)

years_available = sorted(df["year"].unique())
selected_year = st.slider("지도 표시 연도 선택", min_value=years_available[0],
                          max_value=years_available[-1], value=years_available[-1])

map_df = df[df["year"] == selected_year].copy()
fig = px.choropleth_mapbox(
    map_df,
    geojson=gj,
    color="extinction_index",
    color_continuous_scale=COLOR_SCALE,
    range_color=(COLOR_BINS[0], COLOR_BINS[-1]),
    locations="region",
    featureidkey="properties.name",
    mapbox_style="carto-positron",
    zoom=4.8,
    center={"lat": 36.5, "lon": 127.8},
    opacity=0.85,
    labels={"extinction_index": "인구소멸지수"},
    hover_data={"region": True, "extinction_index":":.2f", "year": True}
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

df_line = df[["year", "region", "total_pop"]].copy()
if smooth_window > 1:
    df_line = df_line.sort_values(["region", "year"])
    df_line["total_pop"] = (
        df_line.groupby("region")["total_pop"]
        .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
    )
df_line["표시값"] = scale_values(df_line["total_pop"], unit_scale)
fig2 = px.line(df_line, x="year", y="표시값", color="region",
               labels={"year": "연도", "표시값": f"총인구({unit_scale})", "region": "행정구역"},
               markers=True)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### 📄 전처리된 표 (CSV 다운로드)")
out_df = df[["year","region","total_pop","young_female","old_65plus","extinction_index"]].copy()
out_df = out_df.rename(columns={"year": "date"})
csv = out_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("CSV 다운로드", csv, "processed_public.csv", "text/csv")
st.dataframe(out_df, use_container_width=True)

st.caption("데이터 출처: 통계청 KOSIS OpenAPI | 지도 경계: southkorea-maps (kostat/2013)")
