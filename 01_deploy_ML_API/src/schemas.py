"""Pydantic 스키마"""

from pydantic import BaseModel


# 사용자가 예측을 요청할 때 보내는 입력값(모델 입력 특성)을 정의
class FastasyAcquistionFeatures(BaseModel):
    waiver_value_tier: int
    fantasy_regular_season_weeks_remaining: int
    league_budget_pct_remaining: int


# 모델이 반환할 예측 결과(각 백분위수별 예측값)를 정의
class PredictionOutput(BaseModel):
    winning_bid_10th_percentile: float
    winning_bid_50th_percentile: float
    winning_bid_90th_percentile: float
