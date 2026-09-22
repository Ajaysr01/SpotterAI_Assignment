# Freight Rate Prediction - Machine Learning Assessment Report

## 1. Initial Analysis & Structural Discrepancies
Upon analyzing the provided `train-test.csv`, `validation.csv`, and `december-chart-inputs.csv` files, several critical data-quality issues and structural mismatches were identified:
*   **Static December Predictions:** The initial output for December resulted in a flat line (same price every day). Analysis revealed that `december-chart-inputs.csv` was missing crucial macroeconomic indicators: `market_index` and `quote_signal`, as well as geographical coordinates (`pickup_lat`, `pickup_lon`, etc.). Without these fluctuating variables, a model will naturally output static predictions.
*   **Pricing Anomalies:** By calculating a `rate_per_mile` feature (`posted_rate` / `distance`), extreme outliers were discovered, including rates under $0.50/mile and over $10.00/mile.
*   **Missing Values:** The training data contained missing values in `weight` (300 rows) and `market_index` (374 rows).

## 2. Data Cleaning & Preprocessing
*   **Why it was needed:** Machine learning models are highly sensitive to extreme outliers and cannot process `NaN` (missing) values mathematically in traditional linear models without imputation.
*   **How it was fixed:** 
    *   Filtered out 152 extreme pricing anomalies (keeping `rate_per_mile` between $0.50 and $10.00). This prevented the model from learning from noise or data-entry errors.
    *   Applied median imputation for missing `weight` and `market_index` values, ensuring robust handling of missing data without skewing distributions.

## 3. Feature Engineering
*   **Why it was needed:** Freight markets are highly seasonal, and string-based city names are difficult for tree-based models to parse effectively without massive one-hot encoding schemas that lead to overfitting.
*   **How it was fixed:** 
    *   **Temporal Features:** Extracted `month`, `day_of_week`, `day_of_month`, and an `is_weekend` flag from the transaction date. This helped the model capture retail peak seasons and weekend premiums.
    *   **Spatial Features:** Replaced categorical city names with precise latitude and longitude coordinates. For the December dataset (which was missing these coordinates), the pipeline dynamically mapped and merged the exact `lat`/`lon` from the historical training dataset based on the city strings.

## 4. Cross-Validation & Splitting Strategy
*   **Why it was needed:** Traditional random shuffling (e.g., standard `train_test_split`) causes "data leakage" in logistics data, where the model essentially looks into the future to predict the past, leading to falsely high accuracy during training but poor real-world inference.
*   **How it was fixed:** Implemented a strict **Time-Series Split (80/20)**. The model was trained entirely on older chronological data (January - August 2025) and validated purely on unseen future data (September - October 2025). This guaranteed that the validation metrics accurately represent real-world forecasting capabilities.

## 5. Overcoming the December Inference Mismatch
*   **Why it was needed:** Because the December inference file completely lacked `market_index` and `quote_signal`, simply imputing them with a static mean would result in the flat-line prediction chart we initially observed. 
*   **How it was fixed:** Implemented a two-step modeling pipeline. First, a **Ridge Regression** model was trained on the previous 60 days of data to dynamically forecast what the `market_index` and `quote_signal` would likely be on each specific day in December (based on day of the week and month trends). 
*   **The Result:** By injecting these forecasted economic trends into the December dataset, our final pricing model was able to output realistic, dynamically fluctuating rates across the month.

## 6. Modeling & Final Performance
*   **Why it was needed:** We needed to capture the highly non-linear, multiplicative relationships in freight pricing (e.g., how the combination of distance, specific equipment types, and market urgency affect the final rate).
*   **How it was fixed:** 
    1.  **Baseline:** Trained a standard Linear Regression model to establish a performance floor. (Result: MAE = $132.94).
    2.  **Advanced Model:** Trained a **HistGradientBoostingRegressor** (Scikit-Learn's native equivalent to LightGBM). This model perfectly handles non-linear relationships and categorical features natively.
*   **Final Result:** The Gradient Boosting model achieved a highly accurate **Mean Absolute Error (MAE) of $111.55** and an **R² Score of 0.85** on the unseen test set. The top 5 driving factors for predictions were `distance`, `market_index`, `quote_signal`, `weight`, and `equipment_Flatbed`.
