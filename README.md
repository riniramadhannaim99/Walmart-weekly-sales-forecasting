# Walmart Weekly Sales Forecasting

Store-level prediction of weekly sales for 45 Walmart stores (Feb 2010 to Oct 2012) using a time-based train-test split, a Store-Mean baseline, and tuned Extra Trees.

**Final result (test set, Nov 2011 to Oct 2012):** WAPE 7.2%, MAE about 76,919 and RMSE about 143,586 per store per week. On the log1p scale, MAE 0.07 and R² 0.97.

## Project Overview

The goal is to predict `Weekly_Sales` per store (regression). EDA comes first and decides how the data is prepared. Because the data is a time series, the split is by date instead of random, so the model is trained on the past and tested on the future.

## Dataset

- **Source:** Kaggle, *Walmart Dataset (Retail)*: https://www.kaggle.com/datasets/rutuspatel/walmart-dataset-retail
- **Size:** 6,435 rows, 45 stores, weekly data from 2010-02-05 to 2012-10-26
- **Columns:** `Store`, `Date`, `Weekly_Sales` (target), `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
- The dataset has no store size, location or promotion information.

Download `Walmart.csv` from Kaggle and place it in the same folder as the notebook.

## Method

1. **Cleaning and EDA:** no missing values or duplicates; `log1p` transform of the target to reduce skewness; outliers kept (regional unemployment, store-level spikes).
2. **Feature engineering:** `Year`, `Month`, `Week`, and the holiday flag split into `super_bowl`, `labor_day`, `thanksgiving`, `25_december`.
3. **Time-based split:** train before 2011-11-01 (4,095 rows), test after (2,340 rows).
4. **Baselines:** global mean and Store-Mean (computed from training data only).
5. **Pipeline:** one-hot encoding for `Store`, standard scaling for numeric and time features, wrapped in a scikit-learn `Pipeline`.
6. **Models:** Ridge, Random Forest, Extra Trees, HistGradientBoosting.
7. **Validation and tuning:** `TimeSeriesSplit` (4 folds) and `RandomizedSearchCV` on Extra Trees.
8. **Evaluation and interpretation:** MAE, RMSE, R², WAPE, residual analysis by holiday, impurity-based and permutation feature importance.

## Results

Test set, log1p scale:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (global mean) | 0.501 | 0.588 | -0.003 |
| Store-Mean baseline | 0.097 | 0.141 | 0.942 |
| Ridge | 0.149 | 0.180 | 0.906 |
| Random Forest | 0.096 | 0.134 | 0.948 |
| HistGradientBoosting | 0.092 | 0.124 | 0.955 |
| **Extra Trees (before tuning)** | **0.065** | **0.091** | **0.976** |

Cross-validation (4 time-ordered folds) gives the same ranking. Tuned Extra Trees reaches CV MAE 0.085.

Final model, original scale: **MAE 76,919 | RMSE 143,586 | WAPE 7.2%**.

Error by period (test set):

| Period | WAPE | Bias |
|---|---|---|
| Thanksgiving | 22.2% | +22.1% (underestimate) |
| 25 December | 10.5% | -5.0% |
| Super Bowl | 8.4% | +8.1% |
| Non-holiday | 6.7% | +3.3% |
| Labor Day | 6.3% | +5.1% |

## Key Findings

- **Store identity is the main driver of sales.** Store averages differ about 8x, the Store-Mean baseline alone gives R² 0.942, and Store holds about 96% of the feature importance.
- **Extra Trees improves on the baseline by about 33% in MAE.** Ridge did not beat the baseline.
- **Thanksgiving week is the main source of error.** The model underestimates it by about 22%, which explains why RMSE is about 1.9x MAE.

## Limitations

- Short history (about 2.7 years); each holiday is one week (45 rows) in the test set, so holiday results are indicative only.
- The test set was viewed while comparing models, before cross-validation.
- Weather and economic variables of the forecast period are treated as known; no lag features are used.
- Feature importance is not causal.
- Only Extra Trees was tuned; no prediction intervals.

## Tools

Python 3, Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
jupyter notebook walmart_data_science.ipynb
```

Place `Walmart.csv` in the same folder, then run all cells from top to bottom.

## Repository Structure

```
.
├── walmart_data_science.ipynb   # full analysis
├── Walmart.csv                  # dataset (download from Kaggle)
└── README.md
```

## References

- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Geurts, Ernst and Wehenkel (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3-42.
- Breiman (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
