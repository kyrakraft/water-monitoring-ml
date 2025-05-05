HackerEarth Machine Learning Challenge: Water Consumption Prediction
## Smart Water Monitoring Systems – Household Water Consumption Prediction

This project centered around building a machine learning model that predicts daily water consumption for individual households based on using timestamped, structured data reflecting historical usage patterns, household characteristics, weather conditions, and conservation behaviors. My approach combined exploratory data anlysis, data cleaning, feature engineering, and an XGBoost model with 5-fold cross-validation. Below is a summary.

- Missing values were addressed through median imputation for numeric features like temperature and humidity.
- I flagged suspicious values – such as negative entries in fields like `Residents`, `Guests`, `Water_Price`, and `Humidity` – with binary indicators and replaced them with median values to maintain data quality without discarding information.
- Categorical variables were one-hot encoded.
- Income levels were mapped to an ordinal scale, and invalid values were assigned a default ordinal of -1.
- Timestamps were parsed into month, hour, and day of the week to enable temporal modeling. From these, I created a range of engineered features that reflected behavioral interactions. for example, whether a household was in a high-income bracket during hot weather (`high_temp_x_income`), or whether it had a swimming pool during spring or summer (`warm_with_pool`). Other examples of engineered features included weekend-specific behavior, and seasonal interactions with amenities and other features.
- I did not incorporate lag-based features because the dataset lacked continuous daily records for uniquely identifiable households, making it impossible to track or model temporal behavior of a single household over time.

The model itself was an XGBoost regressor, tuned with early stopping and regularization parameters. I used 5-fold cross-validation to assess performance, reporting a HackerEarth custom score defined as `100 - RMSE`. Because I completed this project after the HackerEarth challenge had ended, model scores on the original HackerEarth test set were unavailable, so I created my own holdout set from the training data, used 5-fold CV on the rest of the training data, and tested my model on this holdout set at the end. The final model resulted in a `100 - RMSE` score of 88.3739.
