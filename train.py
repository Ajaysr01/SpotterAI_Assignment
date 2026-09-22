import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import datetime
import os
from fpdf import FPDF

# --- 1. Data Exploration & Cleaning ---
print("Loading data...")
df = pd.read_csv('train-test.csv')
df['date'] = pd.to_datetime(df['date'])

# Clean outliers (rate per mile)
df['rate_per_mile'] = df['posted_rate'] / df['distance'].replace(0, np.nan)
initial_len = len(df)
df = df[(df['rate_per_mile'] >= 0.5) & (df['rate_per_mile'] <= 10.0)].copy()
print(f"Removed {initial_len - len(df)} anomalies based on rate_per_mile.")

# --- 2. Feature Engineering ---
def engineer_features(data):
    d = data.copy()
    if 'date' not in d.columns:
        return d
    d['date'] = pd.to_datetime(d['date'])
    d['month'] = d['date'].dt.month
    d['day_of_week'] = d['date'].dt.dayofweek
    d['is_weekend'] = (d['day_of_week'] >= 5).astype(int)
    d['day_of_month'] = d['date'].dt.day
    return d

df = engineer_features(df)
df = df.sort_values('date').reset_index(drop=True)

df['weight'] = df['weight'].fillna(df['weight'].median())
df['market_index'] = df['market_index'].fillna(df['market_index'].median())

# --- 3. Cross-Validation & Splitting Strategy ---
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

features = ['distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 
            'market_index', 'quote_signal', 'month', 'day_of_week', 'is_weekend', 'equipment']
num_features = ['distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 
                'market_index', 'quote_signal', 'month', 'day_of_week', 'is_weekend']
cat_features = ['equipment']
target = 'posted_rate'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# --- 4. Modeling ---
print("\n--- Training Baseline Linear Regression ---")
lr_model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
print(f"LR MAE: {lr_mae:.2f}")

print("\n--- Training Advanced Gradient Boosting Regressor ---")
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42))
])
gb_model.fit(X_train, y_train)

# --- 5. Evaluation ---
gb_preds = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_preds)
gb_r2 = r2_score(y_test, gb_preds)
print(f"\n[Gradient Boosting Performance on Test Set]\nMAE: {gb_mae:.2f}\nR2 Score: {gb_r2:.2f}")

# --- Predict for December Data ---
print("\n--- Handling December Data ---")
dec_df = pd.read_csv('december-chart-inputs.csv')
dec_df = engineer_features(dec_df)

pickups = df[['pickup', 'pickup_lat', 'pickup_lon']].drop_duplicates(subset=['pickup'])
deliveries = df[['delivery', 'delivery_lat', 'delivery_lon']].drop_duplicates(subset=['delivery'])
dec_df = dec_df.merge(pickups, on='pickup', how='left')
dec_df = dec_df.merge(deliveries, on='delivery', how='left')

recent_df = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=60))]
X_recent = recent_df[['month', 'day_of_month', 'day_of_week', 'is_weekend']]
model_mi = Ridge().fit(X_recent, recent_df['market_index'])
model_qs = Ridge().fit(X_recent, recent_df['quote_signal'])

X_dec = dec_df[['month', 'day_of_month', 'day_of_week', 'is_weekend']]
dec_df['market_index'] = model_mi.predict(X_dec)
dec_df['quote_signal'] = model_qs.predict(X_dec)

dec_preds = gb_model.predict(dec_df[features])

orig_dec_df = pd.read_csv('december-chart-inputs.csv')
orig_dec_df['predicted_rate'] = dec_preds
orig_dec_df.to_csv('december-chart-inputs.csv', index=False)
print("Updated december-chart-inputs.csv with forecasted features.")

# --- Predict Validation Data ---
print("\n--- Predicting Validation Data ---")
val_df = pd.read_csv('validation.csv')
val_df = engineer_features(val_df)
val_preds = gb_model.predict(val_df[features])

val_out = pd.DataFrame({'load_id': val_df['load_id'], 'predicted_rate': val_preds})
val_out.to_csv('validation_predictions.csv', index=False)
print("Saved validation_predictions.csv")

# --- 6. Generate PDF Report dynamically ---
print("\n--- Generating PDF Report ---")
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Machine Learning Engineer Assessment Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 11)

body = f"""1. Data Exploration & Anomaly Handling
- Identified missing features (market_index and quote_signal) in the December input file.
- Removed extreme anomalies based on unrealistic rate_per_mile (< $0.50 or > $10.00).
- Imputed missing values in weight and market_index using median strategy.

2. Cross-Validation Strategy
- Avoided random shuffling to prevent data leakage.
- Used an 80/20 Time-Series split. The training data was chronologically filtered to train on the oldest 80% and evaluate on the latest 20%.

3. Feature Engineering & Handling Inference Mismatch
- Extracted temporal features: month, day_of_week, is_weekend, and day_of_month.
- Extracted missing spatial features (pickup_lat/lon) by joining city names from the historical dataset.
- Trained a Ridge Regression model to dynamically forecast the missing market_index and quote_signal values for December, which allowed our model to output realistic, fluctuating prices across the month.

4. Modeling & Performance
- Baseline Linear Regression achieved an MAE of ${lr_mae:.2f}.
- Advanced Gradient Boosting (HistGradientBoostingRegressor) achieved an MAE of ${gb_mae:.2f} (R2 = {gb_r2:.2f}).
"""
for line in body.split('\n'):
    pdf.multi_cell(0, 7, line)
    
pdf.ln(5)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "December Forecast Chart", 0, 1)

if os.path.exists('scorer_results/candidate_december.png'):
    pdf.image('scorer_results/candidate_december.png', x=10, w=180)
else:
    pdf.multi_cell(0, 10, "Note: Run score.py to generate scorer_results/candidate_december.png")

pdf.output('report.pdf')
print("report.pdf generated successfully.")
