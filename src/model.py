import os
import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import optuna

print("Model script starting...")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_data_folder = os.path.join(project_root, 'data')

#Load data
#train = pd.read_csv(os.path.join(path_to_data_folder, 'train.csv'))
#test = pd.read_csv(os.path.join(path_to_data_folder, 'test.csv')) 

full_data = pd.read_csv(os.path.join(path_to_data_folder, 'train.csv'))

#calling my holdout set "test"
train, test = train_test_split(full_data, test_size=0.15, random_state=42)


test_timestamps = test['Timestamp']


#FILL MISSING VALUES
train['Temperature'] = train['Temperature'].fillna(train['Temperature'].median())
train['Humidity'] = pd.to_numeric(train['Humidity'], errors='coerce')
train['Humidity'] = train['Humidity'].fillna(train['Humidity'].median())
train['Apartment_Type'] = train['Apartment_Type'].fillna('Unknown')
train['Amenities'] = train['Amenities'].fillna('Unknown')
train['Appliance_Usage'] = train['Appliance_Usage'].fillna(-1)

test['Temperature'] = test['Temperature'].fillna(train['Temperature'].median())
test['Humidity'] = pd.to_numeric(test['Humidity'], errors='coerce')
test['Humidity'] = test['Humidity'].fillna(train['Humidity'].median())
test['Apartment_Type'] = test['Apartment_Type'].fillna('Unknown')
test['Amenities'] = test['Amenities'].fillna('Unknown')
test['Appliance_Usage'] = test['Appliance_Usage'].fillna(-1)


valid_income_levels = ["Low", "Middle", "Upper Middle", "Rich"]
train['Income_Level'] = train['Income_Level'].apply(lambda x: x if x in valid_income_levels else "Unknown")
test['Income_Level'] = test['Income_Level'].apply(lambda x: x if x in valid_income_levels else "Unknown")

income_map = {"Low": 0, "Middle": 1, "Upper Middle": 2, "Rich": 3, "Unknown": -1}
train['Income_Level_Ordinal'] = train['Income_Level'].map(income_map)
train.drop(columns=['Income_Level'], inplace=True)
test['Income_Level_Ordinal'] = test['Income_Level'].map(income_map)
test.drop(columns=['Income_Level'], inplace=True)


suspicious_cols_flag = ['Residents', 'Water_Price', 'Guests', 'Humidity'] 
for col in suspicious_cols_flag:
    train[f'{col}_suspicious_flag'] = (pd.to_numeric(train[col], errors='coerce') < 0).astype(int)

suspicious_cols_to_median = ['Residents', 'Water_Price', 'Guests', 'Humidity'] 
for col in suspicious_cols_to_median:
    median_val = pd.to_numeric(train.loc[train[col] >= 0, col], errors='coerce').median()
    train.loc[train[col] < 0, col] = median_val

for col in suspicious_cols_flag:
    test[f'{col}_suspicious_flag'] = (pd.to_numeric(test[col], errors='coerce') < 0).astype(int)

for col in suspicious_cols_to_median:
    median_val = pd.to_numeric(train.loc[train[col] >= 0, col], errors='coerce').median()
    test.loc[test[col] < 0, col] = median_val

#one hot encode categorical vars
train = pd.get_dummies(train, columns=['Apartment_Type', 'Amenities'], drop_first=False)
test = pd.get_dummies(test, columns=['Apartment_Type', 'Amenities'], drop_first=False)



train.drop(columns=[
    'Apartment_Type_Unknown',
    'Amenities_Unknown'
], inplace=True)


test.drop(columns=[
    'Apartment_Type_Unknown',
    'Amenities_Unknown'
], inplace=True)


train = train.astype({col: 'int' for col in train.columns if col.startswith('Apartment_Type_') or col.startswith('Amenities_')})
test = test.astype({col: 'int' for col in test.columns if col.startswith('Apartment_Type_') or col.startswith('Amenities_')})



#FEATURE ENGINEERING
train['high_temp_x_income'] = train['Temperature'] * (
    train['Income_Level_Ordinal'].isin([3, 4]).astype(int)
)

train['Timestamp'] = pd.to_datetime(train['Timestamp'], format='%d/%m/%Y %H')

#train['Year'] = train['Timestamp'].dt.year
train['Month'] = train['Timestamp'].dt.month
#train['Day'] = train['Timestamp'].dt.day
train['Hour'] = train['Timestamp'].dt.hour
train['DayOfWeek'] = train['Timestamp'].dt.dayofweek + 1

train = train.drop('Timestamp', axis=1)

#define seasons based on month
train['is_spring'] = train['Month'].isin([3, 4, 5]).astype(int)
train['is_summer'] = train['Month'].isin([6, 7, 8]).astype(int)
train['is_fall']   = train['Month'].isin([9, 10, 11]).astype(int)
train['is_winter'] = train['Month'].isin([12, 1, 2]).astype(int)

train['warm_with_pool'] = (
    ((train['is_summer'] == 1) | (train['is_spring'] == 1)) & 
    (train['Amenities_Swimming Pool'] == 1)
).astype(int)

train['cold_with_jacuzzi'] = (
    ((train['is_fall'] == 1) | (train['is_winter'] == 1)) & 
    (train['Amenities_Jacuzzi'] == 1)
).astype(int)


train['warm_with_garden'] = (
    ((train['is_spring'] == 1) | (train['is_summer'] == 1)) &
    (train['Amenities_Garden'] == 1)
).astype(int)


train['weekend_pool_hot'] = (
    (train['DayOfWeek'].isin([6, 7])) &  # Saturday = 6, Sunday = 7
    (train['Amenities_Swimming Pool'] == 1) &
    (train['Temperature'] > 27)
).astype(int)


test['high_temp_x_income'] = test['Temperature'] * (
    test['Income_Level_Ordinal'].isin([3, 4]).astype(int)
)

test['Timestamp'] = pd.to_datetime(test['Timestamp'], format='%d/%m/%Y %H')
test['Month'] = test['Timestamp'].dt.month
test['Hour'] = test['Timestamp'].dt.hour
test['DayOfWeek'] = test['Timestamp'].dt.dayofweek + 1

test['is_spring'] = test['Month'].isin([3, 4, 5]).astype(int)
test['is_summer'] = test['Month'].isin([6, 7, 8]).astype(int)
test['is_fall']   = test['Month'].isin([9, 10, 11]).astype(int)
test['is_winter'] = test['Month'].isin([12, 1, 2]).astype(int)

test['warm_with_pool'] = (
    ((test['is_summer'] == 1) | (test['is_spring'] == 1)) & 
    (test.get('Amenities_Swimming Pool', 0) == 1)
).astype(int)

test['cold_with_jacuzzi'] = (
    ((test['is_fall'] == 1) | (test['is_winter'] == 1)) & 
    (test.get('Amenities_Jacuzzi', 0) == 1)
).astype(int)

test['warm_with_garden'] = (
    ((test['is_spring'] == 1) | (test['is_summer'] == 1)) &
    (test.get('Amenities_Garden', 0) == 1)
).astype(int)

test['weekend_pool_hot'] = (
    (test['DayOfWeek'].isin([6, 7])) &
    (test.get('Amenities_Swimming Pool', 0) == 1) &
    (test['Temperature'] > 27)
).astype(int)



#MODELING
X = train.drop(columns=['Water_Consumption']) 
y = train['Water_Consumption']

X = X.drop(columns=['Timestamp'], errors='ignore')
test = test.drop(columns=['Timestamp'], errors='ignore')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
scores = []
r2_scores = []

def custom_score(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return max(0, 100 - rmse)



for train_idx, val_idx in kf.split(X):
    print(f"Training fold {fold}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(n_estimators=2300, learning_rate=0.01, random_state=42, early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)

    score = custom_score(y_val, preds)
    print(f"Fold {fold} (100 - RMSE): {score:.4f}")
    scores.append(score)

    r2 = r2_score(y_val, preds)
    print(f"Fold {fold} R²: {r2:.4f}")
    r2_scores.append(r2)

    fold += 1

print(f"\nAverage (100 - RMSE) across folds: {np.mean(scores):.4f}")
print(f"\nAverage R² across folds: {np.mean(r2_scores):.4f}")

print("Training final model on full data...")
model = xgb.XGBRegressor(
        n_estimators=2300,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        n_jobs=-1
    )

model.fit(X, y)

# Predict on test set
X_test = test.drop(columns=['Water_Consumption'])
y_test = test['Water_Consumption']


final_preds = model.predict(X_test)

rmse = 100 - np.sqrt(mean_squared_error(y_test, final_preds))
r2 = r2_score(y_test, final_preds)

print(f"Holdout 100 - RMSE: {rmse:.4f}")
print(f"Holdout R²: {r2:.4f}")

""" 
#SUBMIT
submission = pd.DataFrame({
    'Timestamp': test_timestamps,
    'Water_Consumption': final_preds
})
submission.to_csv('../submission/submission.csv', index=False)
"""

print("done")