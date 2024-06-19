import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingRegressor

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor



df=pd.read_csv('/Users/arshadahmed/Desktop/task2sigma/Alpha2.csv')



#*****************encode************************************

def target_encode(df, target, column):
    target_means = df.groupby(column)[target].mean()
    return target_means

cat_columns = ['seller_country', 'seller_badge','product_category', 'clothing_type','region','product_color','product_condition','product_material']

encoding_mappings={}
for col in cat_columns:
    encoding_mappings[col]=target_encode(df,'seller_price',col)
    
for col in cat_columns:
    df[col+'_e']=df[col].map(encoding_mappings[col])

selected_columns = [col+'_e' for col in cat_columns]+['seller_price']
df_e=df[selected_columns]


#*****************splitting the data*********************

# Split data into training and validation sets
X = df_e.drop('seller_price', axis=1)
y = df_e['seller_price']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)

null_indices_train=X_train[X_train.isnull().any(axis=1)].index
null_indices_valid=X_valid[X_valid.isnull().any(axis=1)].index

X_train = X_train.drop(index=null_indices_train)
y_train = y_train.drop(index=null_indices_train)

X_valid = X_valid.drop(index=null_indices_valid)
y_valid = y_valid.drop(index=null_indices_valid)

base_models =[
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42))
]

kf=KFold(n_splits=5, shuffle=True, random_state=42)


voting_regressor=StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

voting_regressor.fit(X_train, y_train)

predictions=voting_regressor.predict(X_valid)

#Evaluation
mae = mean_absolute_error(y_valid, predictions)
print(f"Voting Model MAE: {mae:.2f}")
r2 = r2_score(y_valid, predictions)
print(f"R-squared (R2) score: {r2:.4f}")


dump(voting_regressor, 'voting_regressor_model.joblib')
dump(encoding_mappings, 'encoding_mappings.joblib')


#************Test with values*****************


encoding_mappings = load('encoding_mappings.joblib')
voting_regressor = load('voting_regressor_model.joblib')

cat_columns = ['seller_country', 'seller_badge','product_category', 'clothing_type','region','product_color','product_condition','product_material']

input_data = {
    'seller_country': ['Italy'],
    'seller_badge': ['Trusted'],
    'product_category': ['Men Clothing'],
    'clothing_type': ['Coats & Jackets'],
    'region': ['Europe'],
    'product_color': ['Black'],
    'product_condition': ['Never worn'],
    'product_material': ['Cotton']
}

input_df = pd.DataFrame(input_data)

for col in cat_columns:
    input_df[col + '_e'] = input_df[col].map(encoding_mappings[col])

selected_columns = [col + '_e' for col in cat_columns]
input_df = input_df[selected_columns]

predictions = round(voting_regressor.predict(input_df)[0],2)
print(predictions)

