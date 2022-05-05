# Diabetes_prediction

---
importing Libraries
---

---
import numpy as np # linear algebra

import pandas as pd # data processing

# Input data files are available in the read-only "../input/" directory

import os

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold

---


to display all columns and rows:

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);  # to display all columns and rows

pd.set_option('display.float_format', lambda x: '%.2f' % x) # The number of numbers that will be shown after the comma.

---
2. EDA (Exploratory of Data Analysis)¶

df=pd.read_csv('diabetes.csv')

df.head()

