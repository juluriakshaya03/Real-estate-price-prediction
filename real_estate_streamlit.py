import streamlit as st
import pandas as pd
import numpy as np
housing = pd.read_csv("data.csv")
#housing.head()
#housing.info()
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
#print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_test_set['CHAS'].value_counts()
housing = strat_train_set.copy()
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
#imputer.statistics_
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)
#housing_num_tr.shape
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
#rmse
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
#rmse_scores
from joblib import dump, load
dump(model, 'Dragon.joblib')
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 

page_bg_img = '''
<style>
body {
background-image: url("https://www.google.co.in/url?sa=i&url=https%3A%2F%2Fpngtree.com%2Ffree-backgrounds-photos%2Freal-estate-banner&psig=AOvVaw3Cv0vOGHNimPhGidKEJtVc&ust=1714138554306000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCICp4_i93YUDFQAAAAAdAAAAABAR");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

#st.image('image.png',width=1000)
with st.form("my_form"):
 st.title('Dragon Real Estate Business')
 CRIM = st.number_input('CRIM')
 ZN = st.number_input('ZN')
 INDUS = st.number_input('INDUS')
 CHAS = st.number_input('CHAS')
 NOX = st.number_input('NOX')
 RM = st.number_input('RM')
 AGE = st.number_input('AGE')
 DIS = st.number_input('DIS')
 RAD = st.number_input('RAD')
 TAX = st.number_input('TAX')
 PTRATIO = st.number_input('PTRATIO')
 B = st.number_input('B')
 LSTAT = st.number_input('LSTAT')
 if(st.form_submit_button(label="Submit", help=None,on_click=None)):
   features = np.array([[CRIM, ZN , INDUS, CHAS,NOX,
       RM, AGE,  DIS, RAD , TAX ,
       PTRATIO,  B, LSTAT]])
   st.write('RESULT : ',model.predict(features))

#def fun():
 #features = np.array([[CRIM, ZN , INDUS, CHAS,NOX,
 #      RM, AGE,  DIS, RAD , TAX ,
 #      PTRATIO,  B, LSTAT]])
 #st.write(model.predict(features))