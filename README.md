# 🍊 Tomorrow-Jeju — 제주 특산물 가격 예측

> [와썹 AI 모델링 15기] DACON 제주 특산물 가격 예측 대회
> 시계열 패턴과 트리 앙상블을 활용해 제주 주요 특산물의 가격 변동을 예측합니다.

---

## 팀원

| 이름 | GitHub | 역할 |
|------|--------|------|
| 신우철 | shin-wc | 전체 파이프라인 설계, 모델 버전 관리, 핵심 인사이트 도출 |
| 김대원 | kim-dw | 피처 엔지니어링 실험, 모델 다양성 확장, 피처 중요도 분석 |
| 박효준 | park-hj | 초기 EDA, 데이터 구조 파악 및 시각화 |

---

## 대회 정보

- **대회명**: DACON 제주 특산물 가격 예측 AI 경진대회
- **평가 지표**: MAE (Mean Absolute Error)
- **예측 대상**: 감귤(TG), 무(RD), 브로콜리(BC), 양배추(CB), 당근(CR)
- **테스트 기간**: 2023-03-04 ~ 2023-03-31

---

## 디렉토리 구조

```
Tomorrow-Jeju/
├── Jeju_Final_v1.0.ipynb   # 팀 최종 제출 파일
├── 취합_방향성.txt           # 팀원별 기여 및 의견 반영 정리
├── data/                   # 학습 / 테스트 / 제출 데이터
├── shin-wc/                # 신우철 실험 노트북 (v1.0.0 ~ v7.0.0)
├── kim-dw/                 # 김대원 실험 노트북 (GeminiCLI 시리즈)
└── park-hj/                # 박효준 EDA 노트북
```

---

## 최종 모델 요약

| 항목 | 내용 |
|------|------|
| **모델** | CatBoost + LightGBM + XGBoost VotingRegressor |
| **핵심 피처** | year, month, day, week_day, year_month(누적), week_num(누적), holiday, dist_seollal, dist_chuseok |
| **타겟 변환** | TG → sqrt, 비TG → log1p |
| **검증 방법** | 시즌 Hold-Out (2022-03) |
| **이상치 처리** | TG>20000, RD>5000, BC>8000, CB>2300 → 품목 평균 대체 |

---

## 실험 결과

| 버전 | 모델 | Public Score | 비고 |
|------|------|:---:|------|
| v1.0.1 | DNN + 달력 7개 피처 | **658.6** ✅ | 현재 최고점 |
| v4.0.0 | LightGBM + CatBoost | 672.5 | 트리 앙상블 도입 |
| v4.1.0 | 트리 + EMA 피처 | 1170.5 ❌ | EMA freeze 문제 |
| v5.0.0 | 트리 + 명절 거리 | — | dist_seollal/chuseok 추가 |
| v7.0.0 | VotingRegressor (3종) | — | 순수 트리 회귀 최종 정리 |

---

## 핵심 인사이트

1. **단순할수록 강하다** — 달력 7개 피처만 사용한 v1.0.1이 Public 658.6으로 최고점
2. **EMA 피처 금지** — 테스트 기간에 학습 마지막 값으로 freeze → v4.1.0 Public 1170.5 급락
3. **계절성이 지배적 신호** — 품목×월 히트맵에서 월별 가격 패턴이 뚜렷하게 확인됨
4. **명절 거리 피처 유효** — 설날/추석 전후 30일 가격 급등 패턴 존재, leakage 없음
5. **시즌 Hold-Out 검증** — KFold는 미래 데이터 혼입으로 낙관적 → 2022-03 기준 Hold-Out 채택

---

## 실행 방법

```bash
pip install lightgbm catboost xgboost scikit-learn holidays pandas numpy matplotlib seaborn
```

1. `data/` 폴더에 `train.csv`, `test.csv`, `sample_submission.csv` 준비
2. `Jeju_Final_v1.0.ipynb` 순서대로 실행
3. `results/submission_final.csv` 생성 확인 후 제출
