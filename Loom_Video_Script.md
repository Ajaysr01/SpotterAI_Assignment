# Loom Video Presentation Script (Target: 2-3 Minutes)

**[0:00 - 0:20] Introduction & Objective**
"Hi everyone, my name is [Your Name] and I'm walking you through my solution for the Freight Rate Machine Learning Assessment. Our goal was to build a robust model to predict the `posted_rate` for shipments, validate it on unseen data, and project pricing trends for December."

**[0:20 - 1:00] Key Findings & Data Quality Issues**
"During my initial data exploration, I found two major issues. First, by calculating the rate per mile, I identified and removed about 150 extreme pricing anomalies—like loads priced under 50 cents a mile—which would have severely skewed the model. 
Second, and most importantly, I noticed the provided December dataset was entirely missing key factors: geographical coordinates, `market_index`, and `quote_signal`. Without these, any model would just predict a static, flat line for December."

**[1:00 - 1:40] Data Prep & Splitting Strategy**
"To prep the data, I mapped the missing spatial coordinates into the December file using the historical training dataset. I also engineered temporal features like month, day of the week, and weekend flags. 
For validation, I deliberately avoided standard random shuffling. Because freight dynamics are time-dependent, I used a strict chronological Time-Series split—training on the oldest 80% of data and validating on the most recent 20% to prevent future data leakage."

**[1:40 - 2:20] Modeling Approach & The December Solution**
"To fix the flat-line December predictions, I built a two-stage pipeline. First, I trained a Ridge Regression model on the final 60 days of our historical data to dynamically forecast the missing `market_index` and `quote_signal` across the days in December. 
Once those factors were forecasted, I passed the complete dataset into my primary model: a HistGradientBoostingRegressor, which operates exactly like LightGBM. I chose this because it excels at capturing the non-linear, multiplicative relationships between distance, equipment type, and market urgency."

**[2:20 - 3:00] Code Walkthrough & Conclusion**
*(Action: Share screen showing `model_pipeline_v2.py` and the December chart)*
"In the code, you can see the ColumnTransformer handling median imputations, followed by the Ridge forecaster, and finally the Gradient Boosting pipeline. 
This architecture yielded a strong Mean Absolute Error of $111 on unseen validation data, with an R-squared of 0.85. And as you can see in the generated December chart, by forecasting those missing market indicators, our model now correctly outputs realistic, fluctuating pricing trends across the month rather than a flat line. Thank you!"
