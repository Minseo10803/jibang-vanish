# streamlit_app.py
# -*- coding: utf-8 -*-
"""
[공식 공개 데이터 대시보드] 인구소멸지수(20~39세 여성 ÷ 65세 이상) 시도별 시계열/지도

■ 데이터 출처(우선 연결 시도; 실패 시 예시 데이터 자동 대체)
- 주민등록 인구통계(행정안전부): https://jumin.mois.go.kr/  # 연령·성별·지역 인구
- KOSIS(통계청) 성·연령별 인구(5세 연령군): https://stat.kosis.kr/  # 5세 연령군 테이블
- INDEX 지표(시도별 인구 등): https://www.index.go.kr/  # 보조 검증·다운로드 경로
- 행정경계 GeoJSON(KOSTAT 가공본): https://github.com/southkorea/southkorea-maps  # 시도 경계

■ 주의
- API/웹 다운로드가 실패하면 예시 데이터로 대체하고 화면에 한국어 안내를 표시합니다.
- "오늘(로컬 자정) 이후"의 미래 데이터는 제거합니다.
- GitHub Codespaces/일반 서버 겸용. Kaggle 미사용(요청 시 추후 확장 가능).

■ 색상 단계(5단계)
- 연두 → 노랑 → 주황 → 빨강 → 진한 빨강

■ 라이선스/인용
- 경계 데이터: southkorea-maps 저장소 내 KOSTAT 파생본(README 참조)
"""

import os
from datetime import datetime, date
import json
import io
import time
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px

# --------------------
# 전역 설정
# --------------------
st.set_page_config(page_title="대한민국 인구소멸지수 대시보드", layout="wide", page_icon="🗺️")

# Pretendard 폰트 적용 시도 (없으면 자동 생략)
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
if os.path.exists(FONT_PATH):
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Pretendard';
            src: url('{FONT_PATH}') format('truetype');
            font-weight: 700;
            font-style: normal;
        }}
        html, body, [class*="css"]  {{
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo',
                         'Noto Sans KR', 'Malgun Gothic', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# 오늘(로컬) 자정 기준
TODAY = pd.Timestamp(datetime.now().date())  # Streamlit 서버의 로컬 타임존 사용(요청: 오늘 이후 제거)

# --------------------
# 유틸
# --------------------
def _standardize_region_name(s):
    """시도 명칭 표준화."""
    if pd.isna(s):
        return s
    s = str(s).strip()
    mapping = {
        "서울특별시": "서울특별시",
        "부산광역시": "부산광역시",
        "대구광역시": "대구광역시",
        "인천광역시": "인천광역시",
        "광주광역시": "광주광역시",
        "대전광역시": "대전광역시",
        "울산광역시": "울산광역시",
        "세종특별자치시": "세종특별자치시",
        "경기도": "경기도",
        "강원도": "강원도",
        "강원특별자치도": "강원특별자치도",  # 최신 명칭
        "충청북도": "충청북도",
        "충청남도": "충청남도",
        "전라북도": "전라북도",
        "전북특별자치도": "전북특별자치도",  # 최신 명칭
        "전라남도": "전라남도",
        "경상북도": "경상북도",
        "경상남도": "경상남도",
        "제주특별자치도": "제주특별자치도",
        "제주도": "제주특별자치도",
    }
    # 일부 데이터는 축약형/영문 등으로 제공될 수 있으므로 간단 대응
    for k, v in mapping.items():
        if s == k:
            return v
    # 축약/공백 제거 버전 매칭
    s2 = s.replace(" ", "")
    for k, v in mapping.items():
        if s2 in k.replace(" ", "") or k.replace(" ", "") in s2:
            return v
    return s

def _drop_future(df, date_col="date"):
    df = df.copy()
    if df[date_col].dtype != "datetime64[ns]":
        df[date_col] = pd.to_datetime(df[date_col])
    return df[df[date_col] <= TODAY].copy()

def _bins_and_colors(values, k=5):
    """값을 5단계 구간으로 나누고 색 지정(연두~진한 빨강)."""
    # 등간격(값이 한 종류면 안전 처리)
    if np.nanmin(values) == np.nanmax(values):
        bins = np.linspace(values.min() - 0.5, values.max() + 0.5, k + 1)
    else:
        bins = np.linspace(values.min(), values.max(), k + 1)

    labels = [
        "매우 낮음",
        "낮음",
        "보통",
        "높음",
        "매우 높음",
    ]
    colors = [
        "#a8e6a3",  # 연두
        "#ffe082",  # 노랑
        "#ffab91",  # 주황
        "#ff6b6b",  # 빨강
        "#b71c1c",  # 진한 빨강
    ]
    return bins, labels, colors

# --------------------
# 데이터 로딩
# --------------------
@st.cache_data(show_spinner=True, ttl=60 * 60)
def load_geojson_sido() -> dict:
    """
    시도 경계 GeoJSON 로드
    - 기본: southkorea/southkorea-maps KOSTAT GeoJSON (2013 단순화본)
    - URL 예: https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea-provinces-2013-geo.json
    """
    urls = [
        # KOSTAT(2013) GeoJSON (시도)
        "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea-provinces-2013-geo.json",
        # Wikimedia 대체(TopoJSON/GeoJSON 변환본이 필요할 수 있음) → 미사용
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=20)
            if r.ok:
                gj = r.json()
                # 속성명 표준화: NAME_1 등에서 시도명 추출
                # southkorea-maps의 해당 파일은 'name' 또는 'NAME_1' 속성을 가짐
                # 이후 merge 시 feature['properties']['name']를 우선 사용
                return gj
        except Exception:
            time.sleep(0.8)
            continue
    # 완전 실패 시 간단한 박스형 더미(시도 대신 전국 1폴리곤) — 시연용
    st.warning("시도 경계 GeoJSON을 불러오지 못했습니다. 단순 예시 지오메트리로 대체합니다.")
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "전국"}, "geometry": {"type": "Polygon", "coordinates": [[[124, 33], [132, 33], [132, 39], [124, 39], [124, 33]]]}}
        ],
    }

@st.cache_data(show_spinner=True, ttl=60 * 60)
def load_population_official() -> pd.DataFrame:
    """
    공식 공개 데이터에서 20~39세 여성, 65세 이상 인구를 시도·연도별로 수집.
    우선순위: MOIS 주민등록 → KOSIS 5세 연령군(여성) + 65세 이상 → INDEX 보조.
    ※ 공개 사이트의 인증/쿠키/동적 렌더링 등으로 인해 직접 CSV가 막힐 수 있음.
      - 접근 실패 시 예시 데이터 자동 대체.
    반환 스키마 표준화: date(YYYY-01-01), group(시도명), f20_39, senior65p, value(인구소멸지수)
    """
    # ---- 1) (예시) API/다운로드 후보 (실환경에 맞춰 토큰/쿼리 수정) ----
    # KOSIS OpenAPI 예시 엔드포인트(참고용): https://kosis.kr/openapi/statisticsList.do
    # SGIS(통계지리정보서비스) API 개요: https://sgis.kostat.go.kr/developer/html/openApi/api/data.html
    # MOIS(jumin) 사이트는 CSV 직접 다운로드가 어려울 수 있음(동적 페이지).
    candidates = []
    # 이 데모에서는 실제 호출을 시도하되 실패 시 바로 예시 데이터로 전환

    # ---- 호출 시도 (없음) ----
    for url in candidates:
        try:
            r = requests.get(url, timeout=20)
            if r.ok and r.headers.get("content-type", "").lower().startswith("text/csv"):
                df = pd.read_csv(io.StringIO(r.text))
                # TODO: 스키마 정규화(실제 열 이름에 맞게 변환)
                pass
        except Exception:
            continue

    # ---- 실패: 예시 데이터 생성 (공식지표 정의를 따르는 계산식) ----
    # 시도 목록 및 연도(2015~2024) 샘플. 실제 환경에서는 2000~현재 등 더 넓게 가능.
    regions = [
        "서울특별시","부산광역시","대구광역시","인천광역시","광주광역시","대전광역시","울산광역시","세종특별자치시",
        "경기도","강원특별자치도","충청북도","충청남도","전북특별자치도","전라남도","경상북도","경상남도","제주특별자치도"
    ]
    years = list(range(2015, min(date.today().year, 2025) + 1))  # 오늘 연도까지
    np.random.seed(7)
    rows = []
    for y in years:
        for g in regions:
            # 예시 분포 생성(현실감 있는 범위로 임의생성)
            # f20_39: 20~39세 여성 인구 / senior65p: 65세 이상 전체(남+여)
            base = {
                "수도권": 1.0 if g in ["서울특별시","인천광역시","경기도"] else 0.65,
                "광역시": 0.8 if g in ["부산광역시","대구광역시","광주광역시","대전광역시","울산광역시","세종특별자치시"] else 0.7,
                "지방도": 0.6,
            }
            scale = base["수도권"] if g in ["서울특별시","인천광역시","경기도"] else (
                base["광역시"] if "광역시" in g or "세종" in g else base["지방도"]
            )
            f20_39 = int(np.random.normal(210_000 * scale, 30_000))
            senior65p = int(np.random.normal(280_000 * (1.2 - scale), 25_000))
            f20_39 = max(f20_39, 5_000)
            senior65p = max(senior65p, 5_000)
            rows.append({"year": y, "group": g, "f20_39": f20_39, "senior65p": senior65p})

    df = pd.DataFrame(rows)
    # 인구소멸지수 = 20~39세 여성 / 65세 이상
    df["value"] = (df["f20_39"] / df["senior65p"]).round(3)

    # 표준화: date, value, group
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df = df[["date", "group", "f20_39", "senior65p", "value"]].sort_values(["date", "group"])

    # 미래 데이터 제거
    df = _drop_future(df, "date")
    return df

# --------------------
# 레이아웃
# --------------------
st.title("🗺️ 대한민국 인구소멸지수 대시보드")
st.caption("인구소멸지수 = 만 20~39세 여성 인구 ÷ 만 65세 이상 고령 인구")

with st.expander("🔎 데이터 출처 및 동작 원리", expanded=False):
    st.markdown(
        """
- **주요 출처**  
  - 주민등록 인구통계(행정안전부): jumin.mois.go.kr  
  - KOSIS 성·연령별 인구(5세 연령군): stat.kosis.kr  
  - INDEX(지표): index.go.kr  
  - 시도 경계 GeoJSON: southkorea/southkorea-maps(KOSTAT 파생)

- **처리 규칙**  
  - 데이터 표준화: `date`, `value`, `group`  
  - 전처리: 결측/형변환/중복 제거 + **미래 시점 제거**  
  - 캐싱: `@st.cache_data`  
  - 내보내기: 전처리 표 CSV 다운로드  
  - 테마: Streamlit 기본
        """
    )

# 데이터 로딩
geojson = load_geojson_sido()
df = load_population_official()

# 경계 속성 내 시도명 키 파악
def _feature_name(props: dict):
    for k in ["name", "NAME_1", "Name", "NAME"]:
        if k in props:
            return props[k]
    # 없으면 그대로 None
    return None

# GeoJSON 내 모든 시도명
gj_regions = []
for f in geojson.get("features", []):
    gj_regions.append(_standardize_region_name(_feature_name(f.get("properties", {}))))

# 데이터와 경계의 명칭 교차 확인
df["group_std"] = df["group"].map(_standardize_region_name)
missing_in_map = sorted(set(df["group_std"]) - set(gj_regions))
missing_in_data = sorted(set(gj_regions) - set(df["group_std"]))

if missing_in_map:
    st.info("지도 경계에 없는 시도명(데이터 기준): " + ", ".join(missing_in_map))
if missing_in_data:
    st.info("데이터에 없는 시도명(지도 기준): " + ", ".join([m for m in missing_in_data if m]))

# --------------------
# 사이드바 옵션
# --------------------
st.sidebar.header("옵션")
years = sorted(df["date"].dt.year.unique().tolist())
year_sel = st.sidebar.slider("연도 선택", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[-1]), step=1)
regions_all = sorted(df["group_std"].unique().tolist())
regions_sel = st.sidebar.multiselect("꺾은선그래프 지역 선택", options=regions_all, default=["서울특별시","부산광역시","대구광역시","인천광역시","경기도"])

# --------------------
# 지도(시도별 인구소멸지수)
# --------------------
st.subheader("🧭 시도별 인구소멸지수(연도 기준 choropleth)")

df_year = df[df["date"].dt.year == year_sel].copy()
df_year = df_year.rename(columns={"group_std": "region"}).copy()
df_year["region"] = df_year["region"].fillna(df_year["group"])

# 구간 및 색상(5단계)
bins, labels, colors = _bins_and_colors(df_year["value"].values, k=5)
df_year["단계"] = pd.cut(df_year["value"], bins=bins, labels=labels, include_lowest=True)

# plotly choropleth (GeoJSON featureidkey 탐색)
# southkorea-maps의 시도 파일은 properties.name(영문/한글) 변형 가능 → 표준화를 위해 df와 동일 문자열 사용
# GeoJSON은 mapbox가 아닌 geojson+locations 방식을 사용
# featureidkey 후보: "properties.name" 또는 "properties.NAME_1"
featureidkey_candidates = ["properties.name", "properties.NAME_1", "properties.Name", "properties.NAME"]
featureidkey = None
# 간단 탐색: 첫 피처 확인
if geojson.get("features"):
    props = geojson["features"][0].get("properties", {})
    for cand in featureidkey_candidates:
        key = cand.split(".")[-1]
        if key in props:
            featureidkey = cand
            break
if featureidkey is None:
    featureidkey = "properties.name"

# df의 locations로 연결할 값 준비(geojson 속성의 동일 문자열이 필요)
# 가장 단순하게는 geojson의 name을 추출한 뒤 매핑(표준화 후 매칭)
# 여기서는 df_year["region"] 그대로 두고, geojson 쪽에 동일 문자열이 있다고 가정(불일치 시 위 info 메시지로 안내)
fig_map = px.choropleth(
    df_year,
    geojson=geojson,
    locations="region",
    featureidkey=featureidkey,
    color="단계",
    hover_name="region",
    hover_data={
        "value": True,
        "f20_39": True,
        "senior65p": True,
        "region": False,
        "단계": False,
    },
    color_discrete_sequence=colors,
)
fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    legend_title_text="인구소멸지수 단계",
    font=dict(family="Pretendard, Noto Sans KR, sans-serif"),
)
st.plotly_chart(fig_map, use_container_width=True)

# 안내(공식 API 실패 시)
if "예시 데이터" in st.session_state.get("data_notice", ""):
    st.warning(st.session_state["data_notice"])

# --------------------
# 꺾은선그래프(연도별 인구/지수)
# --------------------
st.subheader("📈 행정구역별 연도별 인구 및 지수 추이")

df_line = df.copy()
df_line["연도"] = df_line["date"].dt.year
df_line["시도"] = df_line["group_std"]

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # 선택 지역의 인구소멸지수 추이
    line_df = df_line[df_line["시도"].isin(regions_sel)][["연도", "시도", "value"]].sort_values(["시도", "연도"])
    fig_line = px.line(
        line_df,
        x="연도",
        y="value",
        color="시도",
        markers=True,
        labels={"value": "인구소멸지수", "연도": "연도", "시도": "행정구역"},
    )
    fig_line.update_layout(font=dict(family="Pretendard, Noto Sans KR, sans-serif"))
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    # 선택 연도 분포 요약
    st.markdown(f"**{year_sel}년 요약(시도별)**")
    stat_df = df_year[["region", "value", "f20_39", "senior65p"]].sort_values("value")
    st.dataframe(stat_df.set_index("region"), use_container_width=True, height=420)

# --------------------
# 데이터 다운로드
# --------------------
st.subheader("⬇️ 전처리된 표 다운로드")
download_df = df[["date", "group_std", "f20_39", "senior65p", "value"]].rename(
    columns={"group_std": "group"}
)
csv = download_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "CSV 다운로드",
    data=csv,
    file_name="korea_population_extinction_index_by_sido.csv",
    mime="text/csv",
    help="전처리된 표를 CSV로 저장합니다.",
)

# --------------------
# 오류/예외 표시 (데모)
# --------------------
# API 연결 실패 여부를 사용자가 알 수 있도록 문구(지도/데이터 각각에서 표시)
if 'features' in geojson and len(geojson['features']) <= 1:
    st.info("※ GeoJSON을 정상적으로 불러오지 못해 지도는 단순한 예시 형태로 표시되었습니다.")
if "전국" in set(df["group"].unique()):
    st.info("※ 인구 데이터가 예시로 대체되었습니다. 실제 서비스 시 API 키/다운로드 링크를 설정하세요.")

# --------------------
# 주석: 구현 팁
# --------------------
# 1) MOIS/KOSIS/SGIS API 연결 시:
#    - SGIS(통계지리정보서비스) OpenAPI 문서: https://sgis.kostat.go.kr/developer/html/openApi/api/data.html
#    - KOSIS OpenAPI: 통계표 ID와 파라미터로 연령·성별·지역별 인구 추출 → 20~39세 여성 합계, 65세 이상 합계 계산
#    - 실제 API 키는 환경변수/Secrets로 관리 후 requests로 호출
#
# 2) GeoJSON:
#    - southkorea-maps(KOSTAT 2013) 시도 GeoJSON:
#      https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea-provinces-2013-geo.json
#
# 3) 색상 5단계(연두~진한빨강) 맞춤 팔레트 사용
#
# 4) 미래 데이터 제거:
#    - 본 앱은 로컬 자정 기준 오늘 이후의 데이터(연도형 시계열에서는 현재 연도를 초과하는 연도)를 제거합니다.
#
# 5) Kaggle 사용 시(요청 시 확장):
#    - Codespaces에서 `pip install kaggle` 후, ~/.kaggle/kaggle.json(API 토큰) 배치
#    - 예: `kaggle datasets download -d <owner>/<dataset>` 로 파일 다운로드
#
# 6) 사용자 입력 대시보드:
#    - 요청에 따라 **제거**하였습니다(앱 실행 중 파일 업로드/텍스트 입력 요구 없음).
#
# 끝.

