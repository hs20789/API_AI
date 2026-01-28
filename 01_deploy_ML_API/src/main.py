"""Fantasy 영입 API"""

from fastapi import FastAPI
import onnxruntime as rt
import numpy as np
from schemas import FastasyAcquistionFeatures, PredictionOutput

api_description = """
이 API는 판타지 풋볼에서 선수를 영입하는 데 필요한 비용 범위를 예측한다.

엔드포인트는 다음과 같이 분류된다.

## 분석
API의 상태 정보 제공

## 예측
선수 영입 비용 예측
"""

# ONNX 모델 불러오기
sess_10 = rt.InferenceSession(
    "../model/acquistion_model_10.onnx",
    providers=["CPUExecutionProvider"],
)
sess_50 = rt.InferenceSession(
    "../model/acquistion_model_50.onnx",
    providers=["CPUExecutionProvider"],
)
sess_90 = rt.InferenceSession(
    "../model/acquistion_model_90.onnx",
    providers=["CPUExecutionProvider"],
)

# 모델별 입력 및 출력 이름 추출
input_name_10 = sess_10.get_inputs()[0].name
label_name_10 = sess_10.get_outputs()[0].name
input_name_50 = sess_50.get_inputs()[0].name
label_name_50 = sess_50.get_outputs()[0].name
input_name_90 = sess_90.get_inputs()[0].name
label_name_90 = sess_90.get_outputs()[0].name

app = FastAPI(
    description=api_description,
    title="Fantasy 영입 API",
    version="0.1",
)


@app.get(
    "/",
    summary="Fantasy 영입 API 상태 확인",
    description="""API가 정상적으로 동작 중인지 확인하는 엔드포인트다.
    이 엔드포인트를 통해 서비스 상태를 미리 점검할 수 있다.""",
    response_description="""메시지가 포함된 JSON 레코드.
    API가 실행 중이면 메시지는 '성공'이라고 표시된다.""",
    operation_id="v0_health_check",
    tags=["분석"],
)
def root():
    return {"message": "API 사앹 확인 성공"}


@app.post(
    "/predict/",
    response_model=PredictionOutput,
    summary="선수 영입 비용 예측",
    description="""이 엔드포인트를 사용해 판타지 풋볼에서 선수를 
    영입하는 데 드는 비용 범위를 예측한다.""",
    response_description="""각 백분위수별 예측 금액이 포함된 JSON 레코들르 반환한다.
    이들은 함꼐 선수 영입 비용의 가능한 범위를 제공한다.""",
    operation_id="v0_predict",
    tags=["예측"],
)
def predict(features: FastasyAcquistionFeatures):
    # Pydantic 모델을 NumPy 배열로 변환
    input_data = np.array(
        [
            [
                features.waiver_value_tier,
                features.fantasy_regular_season_weeks_remaining,
                features.league_budget_pct_remaining,
            ]
        ],
        dtype=np.int64,
    )

    # ONNX 추론 실행
    pred_onx_10 = sess_10.run(
        [label_name_10],
        {input_name_10: input_data},
    )[0]
    pred_onx_50 = sess_50.run(
        [label_name_50],
        {input_name_50: input_data},
    )[0]
    pred_onx_90 = sess_90.run(
        [label_name_90],
        {input_name_90: input_data},
    )[0]

    y10 = pred_onx_10.reshape(-1)[0].item()
    y50 = pred_onx_50.reshape(-1)[0].item()
    y90 = pred_onx_90.reshape(-1)[0].item()
    # 예측을 Pydantic 응답 모델로 반환
    return PredictionOutput(
        winning_bid_10th_percentile=round(float(y10), 2),
        winning_bid_50th_percentile=round(float(y50), 2),
        winning_bid_90th_percentile=round(float(y90), 2),
    )
