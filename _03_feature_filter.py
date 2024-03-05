from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

#%% 对风电数据集进行分析
WindDataset=pd.read_csv("data/dataset/dwd/WindDataset.csv")
#提取特征列名
X = WindDataset.drop('Wind_MWh_credit', axis=1)
y = WindDataset['Wind_MWh_credit']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Creating and fitting the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Getting feature importances
feature_importances = rf.feature_importances_

# Creating a DataFrame to display feature importance
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display
print(features_df)

#%% 对光伏数据集进行分析
SolarDataset=pd.read_csv("data/dataset/dwd/SolarDataset.csv")
#提取特征列名
X = SolarDataset.drop('Solar_MWh_credit', axis=1)
y = SolarDataset['Solar_MWh_credit']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Creating and fitting the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Getting feature importances
feature_importances = rf.feature_importances_

# Creating a DataFrame to display feature importance
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display
print(features_df)

