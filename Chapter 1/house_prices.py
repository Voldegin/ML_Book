import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataset = r"F:\VOLD\ML\ML Book\handson-ml-master\datasets"

housing = pd.read_csv(os.path.join(dataset, "housing", "housing.csv"))

housing.info()
print(housing.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

'''Stratified K-Splitting using income category derived from median income'''
from sklearn.model_selection import StratifiedShuffleSplit

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0, 1.5, 3, 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])

print("\n\n")
print(housing["income_cat"].value_counts())

# housing["income_cat"].hist()
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
stratified_train_set = pd.DataFrame()
stratified_test_set = pd.DataFrame()
for train_index, test_index in split.split(housing, housing["income_cat"]):
    # print(train_index)
    # print(test_index)
    stratified_train_set = housing.loc[train_index]
    stratified_test_set = housing.loc[test_index]

for x in (stratified_test_set, stratified_train_set):
    x.drop("income_cat", axis=1, inplace=True)

'''Checking Correlation'''

housing_check = stratified_train_set.copy()

# housing_check.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
# plt.show()

# housing_check.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
#              s=housing["population"]/100,label="population",figsize=(10,7),
#              c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
# plt.show()

corr_matrix = housing_check.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# from pandas.plotting import scatter_matrix

# attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
# scatter_matrix(housing_check[attributes],figsize=(12,8))
# plt.show()

'''Adding New Attributes'''

housing_check["rooms_per_household"] = housing_check["total_rooms"] / housing_check["households"]
housing_check["bedrooms_per_room"] = housing_check["total_bedrooms"] / housing_check["total_rooms"]
housing_check["population_per_household"] = housing_check["population"] / housing_check["households"]

corr_matrix = housing_check.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

'''Cleaning Data'''
from sklearn.impute import SimpleImputer

housing_clean = stratified_train_set.drop("median_house_value", axis=1)
housing_labels = stratified_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing_clean.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

'''Categorical Data conversion'''
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

housing_cat = housing_clean[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(ordinal_encoder.categories_)
print(housing_cat_encoded[:10])
one_hot_encoder = OneHotEncoder()
housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)
print(one_hot_encoder.categories_)
print(housing_cat_1hot.toarray())

'''Using Pipeline and Classes'''

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attributes = attr_adder.transform(housing_clean.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

# housing_num_pipeline = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing_clean)

print("Done")
