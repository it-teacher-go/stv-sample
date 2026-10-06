import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 흥행 예측기",
    layout="wide"
)

st.title("🎬 영화 흥행 예측기")
st.write(
    "영화 정보를 이용해 **총 관객 수(total_audi)**를 예측하는 "
    "다중 회귀 모델입니다."
)


# --------------------------------------------------
# 2. 데이터 주소
# --------------------------------------------------
DAILY_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/refs/heads/main/kobis_daily.csv"
)

MOVIES_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/refs/heads/main/kobis_movies.csv"
)


# --------------------------------------------------
# 3. 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(
        DAILY_URL,
        encoding="utf-8-sig",
        dtype={"영화코드": str}
    )

    movies = pd.read_csv(
        MOVIES_URL,
        encoding="utf-8-sig",
        dtype={"movieCd": str}
    )

    return daily, movies


daily, movies_raw = load_data()


# --------------------------------------------------
# 4. 기준 기간 확인
# --------------------------------------------------
daily["날짜"] = pd.to_datetime(
    daily["날짜"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

start_date = daily["날짜"].min()
end_date = daily["날짜"].max()


# --------------------------------------------------
# 5. 원본 영화별 표 앞 10줄
# --------------------------------------------------
st.subheader("📋 영화별 표")

st.write(
    "영화별 표 `kobis_movies.csv`의 **맨 위 10줄**입니다."
)

st.dataframe(
    movies_raw.head(10),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 6. 변수 선택
# --------------------------------------------------
st.subheader("① 예측에 사용할 변수 고르기")

st.write(
    "총 관객 수를 예측할 때 사용할 정보를 선택하세요. "
    "여러 변수를 선택하면 **다중 회귀 모델**이 됩니다."
)

c1, c2, c3 = st.columns(3)

with c1:
    use_first_scrn = st.checkbox(
        "첫 관측일 스크린수",
        value=True
    )

    use_first_show = st.checkbox(
        "첫 관측일 상영횟수",
        value=True
    )

    use_peak = st.checkbox(
        "성수기 개봉 여부",
        value=True
    )

with c2:
    use_first_week = st.checkbox(
        "첫 주 관객 수",
        value=True
    )

    use_days = st.checkbox(
        "10위권에 머문 일수",
        value=False
    )

    use_genre = st.checkbox(
        "장르",
        value=False
    )

with c3:
    use_nation = st.checkbox(
        "국가",
        value=False
    )

    use_openDt = st.checkbox(
        "개봉일",
        value=False
    )

    use_first_date = st.checkbox(
        "10위권 첫 등장일",
        value=False
    )


# 선택된 변수 모으기
selected_features = []

if use_first_scrn:
    selected_features.append("first_scrn")

if use_first_show:
    selected_features.append("first_show")

if use_peak:
    selected_features.append("peak")

if use_first_week:
    selected_features.append("first_week_audi")

if use_days:
    selected_features.append("days_in_top10")

if use_genre:
    selected_features.append("genre")

if use_nation:
    selected_features.append("nation")

if use_openDt:
    selected_features.append("openDt_num")

if use_first_date:
    selected_features.append("first_date_num")


if len(selected_features) == 0:
    st.warning("예측에 사용할 변수를 하나 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 7. 데이터 준비
# --------------------------------------------------
movies = movies_raw.copy()


# 날짜를 날짜형으로 변환
movies["openDt_date"] = pd.to_datetime(
    movies["openDt"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

movies["first_date_date"] = pd.to_datetime(
    movies["first_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)


# 회귀 모델에서 날짜를 사용할 수 있도록 숫자로 변환
# 2025-01-01을 기준으로 며칠이 지났는지 계산
DATE_BASE = pd.Timestamp("2025-01-01")

movies["openDt_num"] = (
    movies["openDt_date"] - DATE_BASE
).dt.days

movies["first_date_num"] = (
    movies["first_date_date"] - DATE_BASE
).dt.days


# --------------------------------------------------
# 8. 영화코드 순으로 정렬
# --------------------------------------------------
movies = movies.sort_values(
    "movieCd"
).reset_index(drop=True)


# --------------------------------------------------
# 9. 테스트 / 학습 데이터 나누기
#
# 10편마다
# 앞 3편 → 테스트
# 뒤 7편 → 학습
# --------------------------------------------------
position = np.arange(len(movies))

test_mask = (position % 10) < 3

test_df = movies[test_mask].copy()
train_df = movies[~test_mask].copy()


# --------------------------------------------------
# 10. 입력 변수와 목표값
# --------------------------------------------------
X_train = train_df[selected_features]
y_train = train_df["total_audi"]

X_test = test_df[selected_features]
y_test = test_df["total_audi"]


# 목표값에 결측치가 있으면 학습 자체가 불가능
if y_train.isna().any() or y_test.isna().any():
    st.error(
        "total_audi에 비어 있는 값이 있어 모델을 학습할 수 없습니다."
    )
    st.stop()


# --------------------------------------------------
# 11. 숫자형 / 범주형 변수 구분
# --------------------------------------------------
categorical_features = [
    col for col in selected_features
    if col in ["genre", "nation"]
]

numeric_features = [
    col for col in selected_features
    if col not in categorical_features
]


# 숫자형 변수 처리
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# 범주형 변수 처리
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="알 수 없음"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# 선택한 변수에 따라 전처리 구성
transformers = []

if numeric_features:
    transformers.append(
        (
            "numeric",
            numeric_transformer,
            numeric_features
        )
    )

if categorical_features:
    transformers.append(
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers
)


# --------------------------------------------------
# 12. 다중 선형 회귀 모델
# --------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)


model.fit(X_train, y_train)


# --------------------------------------------------
# 13. 테스트 영화 예측
# --------------------------------------------------
pred = model.predict(X_test)


# --------------------------------------------------
# 14. 평가 지표
# --------------------------------------------------
r2 = r2_score(y_test, pred)

mae = mean_absolute_error(y_test, pred)

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)


# --------------------------------------------------
# 15. 데이터 구성 정보
# --------------------------------------------------
st.subheader("② 학습 데이터와 평가 데이터")

m1, m2, m3 = st.columns(3)

m1.metric(
    "학습에 사용한 영화",
    f"{len(train_df):,}편"
)

m2.metric(
    "점수를 평가한 영화",
    f"{len(test_df):,}편"
)

m3.metric(
    "전체 영화",
    f"{len(movies):,}편"
)


st.info(
    f"📅 기준 기간: "
    f"{start_date.strftime('%Y-%m-%d')} ~ "
    f"{end_date.strftime('%Y-%m-%d')}"
)

st.caption(
    "영화코드 순으로 정렬한 뒤, "
    "열 편마다 앞의 세 편을 테스트용으로 떼어 놓고 "
    "나머지 영화로 모델을 학습합니다."
)


# --------------------------------------------------
# 16. 선택한 변수 표시
# --------------------------------------------------
feature_names = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객 수",
    "days_in_top10": "10위권에 머문 일수",
    "genre": "장르",
    "nation": "국가",
    "openDt_num": "개봉일",
    "first_date_num": "10위권 첫 등장일"
}

selected_names = [
    feature_names[x]
    for x in selected_features
]

st.write(
    "**현재 사용한 변수:** "
    + " · ".join(selected_names)
)


# --------------------------------------------------
# 17. 모델 평가 점수
# --------------------------------------------------
st.subheader("③ 학습하지 않은 영화로 평가")

c1, c2, c3 = st.columns(3)

c1.metric(
    "R²",
    f"{r2:.3f}"
)

c2.metric(
    "MAE",
    f"{mae:,.0f}명"
)

c3.metric(
    "RMSE",
    f"{rmse:,.0f}명"
)


st.caption(
    "R²은 1에 가까울수록 실제 관객 수의 차이를 "
    "모델이 잘 설명한다는 뜻입니다. "
    "MAE는 예측이 실제값에서 평균적으로 몇 명 정도 "
    "벗어났는지를 나타냅니다."
)


# --------------------------------------------------
# 18. 테스트 결과표
# --------------------------------------------------
result = test_df[
    [
        "movieCd",
        "movieNm",
        "total_audi"
    ]
].copy()

result["예측 총 관객 수"] = pred

result["오차"] = (
    result["예측 총 관객 수"]
    - result["total_audi"]
)

result["절대 오차"] = result["오차"].abs()


result = result.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객 수"
    }
)


# 보기 좋게 정수로 표시
result["예측 총 관객 수"] = (
    result["예측 총 관객 수"].round().astype(int)
)

result["오차"] = (
    result["오차"].round().astype(int)
)

result["절대 오차"] = (
    result["절대 오차"].round().astype(int)
)


st.subheader("④ 영화별 예측 결과")

st.write(
    "양수 오차는 실제보다 많이 예측한 경우, "
    "음수 오차는 실제보다 적게 예측한 경우입니다."
)

st.dataframe(
    result,
    use_container_width=True,
    hide_index=True,
    column_config={
        "실제 총 관객 수": st.column_config.NumberColumn(
            format="%d명"
        ),
        "예측 총 관객 수": st.column_config.NumberColumn(
            format="%d명"
        ),
        "오차": st.column_config.NumberColumn(
            format="%d명"
        ),
        "절대 오차": st.column_config.NumberColumn(
            format="%d명"
        )
    }
)


# --------------------------------------------------
# 19. 로그 그래프 준비
# --------------------------------------------------
# 실제 예측값은 평가에 그대로 사용하지만,
# 로그 축에서는 0이나 음수를 표시할 수 없음.
# 1000명 미만 예측은 1000명 위치에 붙여 표시.
plot_pred = np.maximum(pred, 1000)

under_1000 = pred < 1000
under_1000_count = int(under_1000.sum())


plot_df = pd.DataFrame(
    {
        "영화코드": test_df["movieCd"],
        "영화명": test_df["movieNm"],
        "실제 총 관객 수": y_test.to_numpy(),
        "원래 예측값": pred,
        "그래프 예측값": plot_pred,
        "1000명 미만": under_1000
    }
)


# --------------------------------------------------
# 20. Plotly 산점도
# --------------------------------------------------
st.subheader("⑤ 실제 관객 수와 예측 관객 수")

st.write(
    "가로축은 **실제 총 관객 수**, "
    "세로축은 **예측한 총 관객 수**입니다."
)

st.write(
    f"📍 예측 결과가 **1,000명보다 작아 그래프 바닥에 표시된 영화는 "
    f"{under_1000_count}편**입니다."
)


fig = go.Figure()


# 일반 예측 영화
normal = plot_df[~plot_df["1000명 미만"]]

fig.add_trace(
    go.Scatter(
        x=normal["실제 총 관객 수"],
        y=normal["그래프 예측값"],
        mode="markers",
        name="예측 영화",
        customdata=np.stack(
            [
                normal["영화명"],
                normal["영화코드"],
                normal["원래 예측값"]
            ],
            axis=-1
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "영화코드: %{customdata[1]}<br>"
            "실제: %{x:,.0f}명<br>"
            "예측: %{customdata[2]:,.0f}명"
            "<extra></extra>"
        )
    )
)


# 1,000명보다 작게 예측된 영화
low = plot_df[plot_df["1000명 미만"]]

if len(low) > 0:
    fig.add_trace(
        go.Scatter(
            x=low["실제 총 관객 수"],
            y=low["그래프 예측값"],
            mode="markers",
            name="예측 1,000명 미만",
            marker=dict(
                symbol="triangle-up",
                size=10
            ),
            customdata=np.stack(
                [
                    low["영화명"],
                    low["영화코드"],
                    low["원래 예측값"]
                ],
                axis=-1
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "영화코드: %{customdata[1]}<br>"
                "실제: %{x:,.0f}명<br>"
                "원래 예측: %{customdata[2]:,.0f}명<br>"
                "그래프에서는 1,000명 위치에 표시"
                "<extra></extra>"
            )
        )
    )


# --------------------------------------------------
# 21. 실제값 = 예측값 대각선
# --------------------------------------------------
positive_actual = y_test[y_test > 0]

axis_min = min(
    positive_actual.min(),
    1000
)

axis_max = max(
    y_test.max(),
    plot_pred.max()
)

fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="실제값 = 예측값",
        hoverinfo="skip"
    )
)


# --------------------------------------------------
# 22. 로그 축 설정
# --------------------------------------------------
fig.update_layout(
    height=650,
    xaxis_title="실제 총 관객 수",
    yaxis_title="예측 총 관객 수",
    hovermode="closest"
)

fig.update_xaxes(
    type="log",
    tickformat=","
)

fig.update_yaxes(
    type="log",
    range=[
        np.log10(1000),
        np.log10(axis_max * 1.2)
    ],
    tickformat=","
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.caption(
    "점이 대각선에 가까울수록 실제 관객 수와 예측 관객 수가 비슷합니다. "
    "세로축에서 1,000명보다 작게 예측된 값은 로그 축에 표시할 수 있도록 "
    "1,000명 위치에 붙여 표시했으며, R²·MAE·RMSE 계산에는 원래 예측값을 사용했습니다."
)
