import json

with open('RE.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Remove the previously appended cells at the end (keep only the original parts)
# The original notebook had roughly 20-21 sections before my last append.
# My last append added 7 cells starting with "📈 [추가 전략]".
new_cells_count = 7
nb['cells'] = nb['cells'][:-new_cells_count]

# 2. Find and update Section 15 (Modeling Setup)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '15. 모델링 준비' in "".join(cell['source']):
        # Found the target markdown, now find the code cell immediately after it
        setup_code_idx = i + 1
        nb['cells'][setup_code_idx]['source'] = [
            "# 1. 외부 데이터(수출입) 연동 및 병합\n",
            "trade = pd.read_csv('international_trade.csv')\n",
            "item_map = {'감귤': 'TG', '당근': 'CR', '양배추': 'CB', '무': 'RD', '꽃양배추와 브로콜리(broccoli)': 'BC'}\n",
            "trade['item'] = trade['품목명'].apply(lambda x: item_map.get(x, 'Other'))\n",
            "trade = trade[trade['item'] != 'Other']\n",
            "trade['year'] = trade['기간'].apply(lambda x: int(x.split('-')[0]))\n",
            "trade['month'] = trade['기간'].apply(lambda x: int(x.split('-')[1]))\n",
            "trade = trade.groupby(['item', 'year', 'month']).sum().reset_index()\n",
            "\n",
            "train = pd.merge(train, trade[['item', 'year', 'month', '수출 중량', '수입 중량']], on=['item', 'year', 'month'], how='left').fillna(0)\n",
            "test = pd.merge(test, trade[['item', 'year', 'month', '수출 중량', '수입 중량']], on=['item', 'year', 'month'], how='left').fillna(0)\n",
            "\n",
            "# 2. TG 전용 전략: 명절 특수성 (설날 근접도 D+N 변수)\n",
            "seollal_dates = pd.to_datetime(['2019-02-05', '2020-01-25', '2021-02-12', '2022-02-01', '2023-01-22'])\n",
            "def get_holiday_dist(date):\n",
            "    diffs = [(date - h).days for h in seollal_dates]\n",
            "    return min(diffs, key=abs)\n",
            "\n",
            "train['dist_to_seollal'] = train['timestamp'].apply(get_holiday_dist)\n",
            "test['dist_to_seollal'] = test['timestamp'].apply(get_holiday_dist)\n",
            "\n",
            "# 3. 그룹별 X, y 준비 및 스케일링\n",
            "# TG 그룹 (수출 데이터 강조)\n",
            "train_tg = train[train['item'] == 'TG'].copy()\n",
            "test_tg = test[test['item'] == 'TG'].copy()\n",
            "tg_drop_cols = ['ID', 'timestamp', 'supply(kg)', 'price(원/kg)', 'target', 'year_month', '수입 중량']\n",
            "X_tg = train_tg.drop(columns=tg_drop_cols)\n",
            "y_tg = train_tg['price(원/kg)']\n",
            "X_test_tg = test_tg.drop(columns=['ID', 'timestamp', 'year_month', '수입 중량'])\n",
            "\n",
            "# Non-TG 그룹 (수입 데이터 강조 + 로그 스케일링)\n",
            "train_non_tg = train[train['item'] != 'TG'].copy()\n",
            "test_non_tg = test[test['item'] != 'TG'].copy()\n",
            "nontg_drop_cols = ['ID', 'timestamp', 'supply(kg)', 'price(원/kg)', 'target', 'year_month', '수출 중량', 'dist_to_seollal']\n",
            "X_nontg = train_non_tg.drop(columns=nontg_drop_cols)\n",
            "y_nontg = np.log1p(train_non_tg['price(원/kg)'])\n",
            "X_test_nontg = test_non_tg.drop(columns=['ID', 'timestamp', 'year_month', '수출 중량', 'dist_to_seollal'])\n",
            "\n",
            "# 범주형 변수 지정\n",
            "cat_tg = ['corporation', 'location', 'year', 'month', 'day', 'day_of_week', 'is_sunday', 'is_holiday']\n",
            "cat_nontg = ['item', 'corporation', 'location', 'year', 'month', 'day', 'day_of_week', 'is_sunday', 'is_holiday']\n",
            "\n",
            "for col in cat_tg: \n",
            "    X_tg[col] = X_tg[col].astype('category')\n",
            "    X_test_tg[col] = X_test_tg[col].astype('category')\n",
            "for col in cat_nontg: \n",
            "    X_nontg[col] = X_nontg[col].astype('category')\n",
            "    X_test_nontg[col] = X_test_nontg[col].astype('category')\n",
            "\n",
            "print('TG / Non-TG 그룹별 모델링 준비 완료 (외부 데이터 및 스케일링 포함)')"
        ]
        break

# 3. Save updated notebook
with open('RE.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
