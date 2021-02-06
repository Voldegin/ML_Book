import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataset = r"F:\VOLD\ML\ML Book\handson-ml-master\datasets"

housing = pd.read_csv(os.path.join(dataset, "housing", "housing.csv"))

housing.info()
print(housing.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

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

print("Done")
