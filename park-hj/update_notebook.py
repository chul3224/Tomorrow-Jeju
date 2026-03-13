import json
import pandas as pd

with open('RE.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define new cells
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📈 [추가 전략] 품목별 맞춤형 피처 엔지니어링 및 그룹별 모델링\n",
            "\n",
            "사용자 요청에 따라 감귤(TG)과 비감귤(Non-TG) 그룹을 분리하고, 각 그룹의 특성에 맞는 외부 데이터(수출입) 및 명절 변수를 추가합니다."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. 외부 데이터(수출입) 연동 및 품목별 피처 할당\n",
            "trade = pd.read_csv('international_trade.csv')\n",
            "item_map = {'감귤': 'TG', '당근': 'CR', '양배추': 'CB', '무': 'RD', '꽃양배추와 브로콜리(broccoli)': 'BC'}\n",
            "trade['item'] = trade['품목명'].apply(lambda x: item_map.get(x, 'Other'))\n",
            "trade = trade[trade['item'] != 'Other']\n",
            "trade['year'] = trade['기간'].apply(lambda x: int(x.split('-')[0]))\n",
            "trade['month'] = trade['기간'].apply(lambda x: int(x.split('-')[1]))\n",
            "trade = trade.groupby(['item', 'year', 'month']).sum().reset_index()\n",
            "\n",
            "# train/test에 병합 (기간이 없는 test의 경우 전월 또는 전년 동월 데이터 활용 가능하나 여기선 merge 후 결측치 처리)\n",
            "train = pd.merge(train, trade[['item', 'year', 'month', '수출 중량', '수입 중량']], on=['item', 'year', 'month'], how='left').fillna(0)\n",
            "test = pd.merge(test, trade[['item', 'year', 'month', '수출 중량', '수입 중량']], on=['item', 'year', 'month'], how='left').fillna(0)\n",
            "\n",
            "print('수출입 데이터 병합 완료')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. TG 전용 전략: 명절 특수성 (설날 전후 D+N 변수)\n",
            "holiday_dates = ['2019-02-05', '2020-01-25', '2021-02-12', '2022-02-01', '2023-01-22']\n",
            "holiday_dates = pd.to_datetime(holiday_dates)\n",
            "\n",
            "def get_holiday_dist(date):\n",
            "    diffs = [(date - h).days for h in holiday_dates]\n",
            "    min_diff = min(diffs, key=abs)\n",
            "    return min_diff\n",
            "\n",
            "train['dist_to_seollal'] = train['timestamp'].apply(get_holiday_dist)\n",
            "test['dist_to_seollal'] = test['timestamp'].apply(get_holiday_dist)\n",
            "\n",
            "print('설날 근접도 변수 생성 완료')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. 그룹 분리 및 Non-TG 가격 분포 확인/스케일링\n",
            "train_tg = train[train['item'] == 'TG'].copy()\n",
            "train_non_tg = train[train['item'] != 'TG'].copy()\n",
            "\n",
            "print('--- Non-TG 품목별 가격 통계 ---')\n",
            "print(train_non_tg.groupby('item')['price(원/kg)'].describe())\n",
            "\n",
            "# Non-TG 품목 간 가격 차이가 크므로 Log Transformation 고려\n",
            "train_non_tg['price_log'] = np.log1p(train_non_tg['price(원/kg)'])\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.histplot(train_non_tg['price_log'], kde=True)\n",
            "plt.title('Non-TG Price (Log Transformed) Distribution')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. 그룹별 개선된 모델링 전략 수행\n",
            "*   **TG 전용 모델**: `수출 중량`과 `설날 근접도`를 핵심 피처로 사용\n",
            "*   **Non-TG 전용 모델**: `수입 중량`을 핵심 피처로 사용하고 로그 스케일 타겟 학습"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from catboost import CatBoostRegressor\n",
            "\n",
            "# TG 모델 학습 (수출 데이터 강조)\n",
            "tg_features = ['corporation', 'location', 'year', 'month', 'day', 'day_of_week', 'is_sunday', 'week_num', 'is_holiday', '수출 중량', 'dist_to_seollal']\n",
            "tg_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=100, cat_features=['corporation', 'location'])\n",
            "tg_model.fit(train_tg[tg_features], train_tg['price(원/kg)'])\n",
            "\n",
            "# Non-TG 모델 학습 (수입 데이터 강조 + 로그 스케일)\n",
            "nontg_features = ['item', 'corporation', 'location', 'year', 'month', 'day', 'day_of_week', 'is_sunday', 'week_num', 'is_holiday', '수입 중량']\n",
            "nontg_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=100, cat_features=['item', 'corporation', 'location'])\n",
            "nontg_model.fit(train_non_tg[nontg_features], train_non_tg['price_log'])\n",
            "\n",
            "print('그룹별 모델 학습 완료')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. 최종 예측 및 후처리\n",
            "test_tg = test[test['item'] == 'TG'].copy()\n",
            "test_non_tg = test[test['item'] != 'TG'].copy()\n",
            "\n",
            "test_tg['ans'] = tg_model.predict(test_tg[tg_features])\n",
            "test_non_tg['ans'] = np.expm1(nontg_model.predict(test_non_tg[nontg_features]))\n",
            "\n",
            "# 결과 합치기\n",
            "test_final = pd.concat([test_tg, test_non_tg]).sort_index()\n",
            "submission['answer'] = test_final['ans']\n",
            "\n",
            "# 일요일 0원 처리\n",
            "submission.loc[submission['ID'].str.contains('20230305|20230312|20230319|20230326'), 'answer'] = 0\n",
            "submission.loc[submission['answer'] < 0, 'answer'] = 0\n",
            "\n",
            "submission.to_csv('improved_submission.csv', index=False)\n",
            "print('개선된 전략 기반 제출 파일 생성 완료 (improved_submission.csv)')"
        ]
    }
]

# Append new cells
nb['cells'].extend(new_cells)

# Save updated notebook
with open('RE.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
