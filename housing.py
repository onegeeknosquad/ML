#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 07:24:30 2018

@author: mrpotatohead
"""
import numpy as np
from get_data import load_housing_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

housing = load_housing_data()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#Create income category attribute
housing["income_cat"] = np.ceil(housing["medium_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

#Stratified Sampling based on the income category
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#strat_test_set["income_cat"].value_counts() / len(strat_test_set)
    
#Remove income_cat so the data is in its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
#Create a copy of the training set for messing around
#housing = strat_train_set.copy()

#Prepare the Data
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()