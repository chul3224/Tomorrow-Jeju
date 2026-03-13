import json

with open('RE.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    # 1. 섹션 12 시각화 코드 개선 및 해석 마크다운 추가
    if cell['cell_type'] == 'markdown' and '12. 공급량(Supply)과 가격(Price)의 상관관계 분석' in "".join(cell['source']):
        code_idx = i + 1
        nb['cells'][code_idx]['source'] = [
            "import matplotlib.ticker as ticker\n",
            "\n",
            "print(\"--- 품목별 공급량-가격 상관계수 ---\")\n",
            "for item in train['item'].unique():\n",
            "    item_df = train[(train['item'] == item) & (train['price(원/kg)'] > 0) & (train['supply(kg)'] > 0)]\n",
            "    corr = item_df['supply(kg)'].corr(item_df['price(원/kg)'])\n",
            "    print(f\"{item}: {corr:.4f}\")\n",
            "\n",
            "plt.figure(figsize=(18, 10))\n",
            "items = train['item'].unique()\n",
            "for i, item in enumerate(items):\n",
            "    ax = plt.subplot(2, 3, i+1)\n",
            "    item_df = train[(train['item'] == item) & (train['price(원/kg)'] > 0) & (train['supply(kg)'] > 0)]\n",
            "    sns.regplot(x='supply(kg)', y='price(원/kg)', data=item_df, \n",
            "                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})\n",
            "    \n",
            "    plt.title(f'{item} 공급량 vs 가격 상관관계')\n",
            "    plt.xscale('log')\n",
            "    plt.yscale('log')\n",
            "    \n",
            "    # 축 단위 보기 쉽게 변경 (10^n -> 일반 숫자)\n",
            "    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x >= 1 else f'{x:.1f}'))\n",
            "    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y):,}' if y >= 1 else f'{y:.1f}'))\n",
            "    \n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
        # 상관관계 해석 마크다운 삽입
        interpretation_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 💡 상관계수 및 산점도 해석 방법\n",
                "\n",
                "1.  **상관계수 ($r$)**: -1에서 1 사이의 값을 가집니다.\n",
                "    *   **음수 (-)**: 한쪽이 커지면 다른 쪽은 작아지는 관계입니다. 농산물은 **\"공급이 늘면 가격이 떨어진다\"**는 경제 원칙에 따라 보통 음수가 나옵니다.\n",
                "    *   **-0.5 이하**: 꽤 뚜렷한 반비례 관계입니다. (예: 감귤(TG)은 공급량 관리가 가격에 매우 중요함)\n",
                "    *   **0에 가까움**: 공급량 외에 다른 요인(수입량, 명절 특수성, 품질 등)이 가격 결정에 더 크게 작용함을 의미합니다.\n",
                "2.  **로그 스케일 그래프**: 데이터가 특정 구간(낮은 가격대)에 너무 몰려 있으면 추세를 보기 어렵기 때문에, 축 간격을 조정하여 전체적인 흐름(빨간 선)을 더 명확히 보기 위해 사용합니다.\n",
                "3.  **빨간 실선**: 공급량에 따른 가격의 평균적인 변화 추세입니다. 이 선이 가파르게 내려갈수록 공급량 변화에 가격이 민감하게 반응하는 품목입니다."
            ]
        }
        nb['cells'].insert(code_idx + 1, interpretation_cell)

    # 2. 섹션 15 코드 아래에 상세 설명 마크다운 추가
    if cell['cell_type'] == 'markdown' and '15. 모델링 준비' in "".join(cell['source']):
        # 섹션 15는 방금 통합했으므로 코드 셀이 i+1 위치에 있음
        desc_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🛠️ 데이터 준비 과정 상세 해석\n",
                "\n",
                "이 단계에서는 분석 결과를 바탕으로 **감귤(TG)**과 **비감귤(Non-TG)**의 특성이 판이하게 다르다는 점을 모델에 반영하기 위해 다음과 같은 전처리를 수행했습니다.\n",
                "\n",
                "1.  **외부 무역 데이터 통합**: `international_trade.csv`를 연동하여 품목별 수출/수입량을 추가했습니다. \n",
                "    *   브로콜리(BC)나 당근(CR)처럼 수입 비중이 높은 품목은 국내 공급량(`supply`)뿐만 아니라 해외 수입량에 의해서도 가격 압박을 받기 때문입니다.\n",
                "2.  **명절 특수성 (설날 근접도) 반영**: 감귤(TG)은 설날 전후 선물용 수요가 폭발적입니다. \n",
                "    *   단순한 날짜 정보 대신 **'설날 당일로부터 며칠 전후인가'**를 나타내는 변수를 생성하여 명절 효과를 모델이 학습하게 했습니다.\n",
                "3.  **그룹별 전용 데이터셋 분리**:\n",
                "    *   **TG 그룹**: 수출 데이터와 명절 근접도를 핵심 피처로 포함합니다. (수입 영향은 적으므로 제외)\n",
                "    *   **Non-TG 그룹**: 수입 데이터와 품목별 특성을 핵심 피처로 포함합니다.\n",
                "4.  **타겟 스케일링 (Log Transformation)**:\n",
                "    *   비감귤 품목들은 서로 가격 편차가 크고 분포가 한쪽으로 치우쳐 있어, **로그 변환**을 통해 데이터 분포를 정규분포 가깝게 만들어 모델의 학습 안정성을 높였습니다.\n",
                "5.  **범주형 변수 처리**: 법인, 지역, 품목 등의 문자열 데이터를 모델이 인식할 수 있는 `category` 타입으로 변환하여 학습 준비를 마쳤습니다."
            ]
        }
        # 현재 i+1이 코드 셀이므로 i+2 위치에 설명 삽입
        nb['cells'].insert(i + 2, desc_cell)

# 결과 저장
with open('RE.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
