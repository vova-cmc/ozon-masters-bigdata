from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split, GridSearchCV

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
cat_features_to_use = ['cf6', 'cf9', 'cf13', 'cf16', 'cf17', 'cf18', 'cf19', 'cf25', 'cf26']

fields = ["id", "label"] + numeric_features + categorical_features

#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer_num', SimpleImputer(strategy='median'))
#    ,('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, cat_features_to_use)
    ]
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #('linearregression', LinearRegression())
    ('model', RandomForestClassifier(max_depth=4, max_features=0.3, verbose=2, n_jobs=4))
])