# Diabetes_prediction

---
importing Libraries
---

---
import numpy as np # linear algebra

import pandas as pd # data processing

Input data files are available in the read-only "../input/" directory

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


To display all columns and rows:

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);  # to display all columns and rows

pd.set_option('display.float_format', lambda x: '%.2f' % x) # The number of numbers that will be shown after the comma.

---
# 2. EDA (Exploratory of Data Analysis)¶

df=pd.read_csv('diabetes.csv')

df.head()

df.shape

df.info() 

*see the column names and its datatypes

df.dtypes

*display the data types

#check for missing values, count them and print the sum for every column

df.isnull().sum() 

#conclusion :- there are no null values in this dataset

#check the basic statistics from given data

df.describe() #helps us to understand how data has been spread across the table.

#count :- the number of NoN-empty rows in a feature.

#mean :- mean value of that feature.

#std :- Standard Deviation Value of that feature.

#min :- minimum value of that feature.

#max :- maximum value of that feature.

#25%, 50%, and 75% are the percentile/quartile of each features.

df.Outcome.value_counts()

df["Age"].hist(edgecolor = "black");

![image](https://user-images.githubusercontent.com/100121721/166847547-d30e6133-bb3a-465d-9bff-0b954f5eaa57.png)

sns.countplot(x="Outcome",data=df,palette="bwr")

![image](https://user-images.githubusercontent.com/100121721/166847573-82beab5c-87cb-47ea-84d7-a4cce3b37406.png)

plt.figure(figsize=(13,6))

g = sns.kdeplot(df["Pregnancies"][df["Outcome"] == 1], color="Red", shade = True)

g = sns.kdeplot(df["Pregnancies"][df["Outcome"] == 0], ax =g, color="Green", shade= True)

g.set_xlabel("Pregnancies")

g.set_ylabel("Frequency")

g.legend(["Positive","Negative"])


![image](https://user-images.githubusercontent.com/100121721/166847631-f9321985-66f3-4044-a335-d7ada3be43c5.png)

df.corr()

#correlation amoung all variables

# heatmap

sns.heatmap(df.corr(), annot=True)

![image](https://user-images.githubusercontent.com/100121721/166847693-d897b27a-cb06-4fcb-ad7a-3cc1064b8b97.png)


DiabetesP=len(df[df.Outcome==1])

DiabetesN=len(df[df.Outcome==0])

print("The % of patients with Diabetes is:",round((DiabetesP/len(df))*100,2))

print("The % of patients without Diabetes is:",round((DiabetesN/len(df))*100,2))

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)

df.isnull().sum()

def carp(x,y):
    
    z = x*y
    
    return z
    
 carp(4,5)
 
 ##The missing values will be filled with the median values of each variable.

def median_target(var):   
    
    temp = df[df[var].notnull()]
    
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    
    return temp

#Distplot

#distribution of frequency in between all the variables and outcome

import warnings

warnings.filterwarnings("ignore")

fig, ax2 = plt.subplots(4, 2, figsize=(16, 16))

sns.distplot(df['Pregnancies'],ax=ax2[0][0])

sns.distplot(df['Glucose'],ax=ax2[0][1])

sns.distplot(df['BloodPressure'],ax=ax2[1][0])

sns.distplot(df['SkinThickness'],ax=ax2[1][1])

sns.distplot(df['Insulin'],ax=ax2[2][0])

sns.distplot(df['BMI'],ax=ax2[2][1])

sns.distplot(df['DiabetesPedigreeFunction'],ax=ax2[3][0])

sns.distplot(df['Age'],ax=ax2[3][1])

#BloodPressure, BMI and Glucose are normally distributed whereas DiabetePedigreeFunction, Age, Pregancies and others are positively skewed.

![image](https://user-images.githubusercontent.com/100121721/166847880-115b2321-40be-4e08-b935-80bb89541f41.png)


plt.figure(figsize=(16,12))

sns.set_style(style='whitegrid')

plt.subplot(3,3,1)

sns.boxplot(x='Glucose',data=df)

plt.subplot(3,3,2)

sns.boxplot(x='BloodPressure',data=df)

plt.subplot(3,3,3)

sns.boxplot(x='Insulin',data=df)

plt.subplot(3,3,4)

sns.boxplot(x='BMI',data=df)

plt.subplot(3,3,5)

sns.boxplot(x='Age',data=df)

plt.subplot(3,3,6)

sns.boxplot(x='SkinThickness',data=df)

plt.subplot(3,3,7)

sns.boxplot(x='Pregnancies',data=df)

plt.subplot(3,3,8)

sns.boxplot(x='DiabetesPedigreeFunction',data=df)

#boxplot to display the outliers in the data

![image](https://user-images.githubusercontent.com/100121721/166847954-06344c61-b582-41fe-bf4c-07e947ee98cd.png)


columns = df.columns

columns = columns.drop("Outcome")

# The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick.

columns = df.columns

columns = columns.drop("Outcome")

for col in columns:
    
    df.loc[(df['Outcome'] == 0 ) & (df[col].isnull()), col] = median_target(col)[col][0]
    
    df.loc[(df['Outcome'] == 1 ) & (df[col].isnull()), col] = median_target(col)[col][1]
    
 df.loc[(df['Outcome'] == 0 ) & (df["Pregnancies"].isnull()), "Pregnancies"]
 
 df[(df['Outcome'] == 0 ) & (df["BloodPressure"].isnull())]
 
 ##outlier analysis
 ---
 
 Q1 = df["BloodPressure"].quantile(0.25)
 
Q3 = df["BloodPressure"].quantile(0.75)

IQR = Q3-Q1

lower = Q1 - 1.5*IQR

upper = Q3 + 1.5*IQR

lower

upper

df[(df["BloodPressure"] > upper)].any(axis=None)

for feature in df:

    print(feature)

for feature in df:
    
    Q1 = df[feature].quantile(0.05)
    
    Q3 = df[feature].quantile(0.95)
    
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5*IQR
    
    upper = Q3 + 1.5*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
    
        print(feature,"yes")
        
    else:
    
        print(feature, "no")

##feature engineering

#According to BMI, some ranges were determined and categorical variables were assigned.

NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")


df["NewBMI"] = NewBMI

df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]

df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]

df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]

df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]

df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]

df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]

def set_insulin(row):

    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
        
    else:
        return "Abnormal"
        
 df["NewInsulinScore"] = df.apply(set_insulin, axis=1)
 
 #Some intervals were determined according to the glucose variable and these were assigned categorical variables.
 
NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")

df["NewGlucose"] = NewGlucose

df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]

df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]

df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]

df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]

##one hot encoding

df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)

categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]
                     
##Feature standardization

y = df["Outcome"]

X = df.drop(["Outcome",'NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)
cols = X.columns

index = X.index

from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(X)

X = transformer.transform(X)

X = pd.DataFrame(X, columns = cols, index = index)

X = pd.concat([X, categorical_df], axis = 1)

##model

models = []

models.append(('LR', LogisticRegression(random_state = 12345)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier(random_state = 12345)))

models.append(('RF', RandomForestClassifier(random_state = 12345)))

models.append(('SVM', SVC(gamma='auto', random_state = 12345)))

models.append(('XGB', GradientBoostingClassifier(random_state = 12345)))

models.append(("LightGBM", LGBMClassifier(random_state = 12345)))

#evaluate each model in turn

results = []

names = []

for name, model in models:
    
        kfold = KFold(n_splits = 10,shuffle=True, random_state = 12345)
        
        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")
        
        results.append(cv_results)
        
        names.append(name)
        
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        
        print(msg)
![image](https://user-images.githubusercontent.com/100121721/166848577-0d386b8b-a569-41e0-9266-e31b7f6e03ac.png)


#boxplot algorithm comparison

fig = plt.figure(figsize=(15,10))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()

![image](https://user-images.githubusercontent.com/100121721/166848633-fc0cb4f5-6007-4465-9084-3450c274d144.png)

##model tuning

lgbm = LGBMClassifier(random_state = 12345)

lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
              "n_estimators": [500, 1000, 1500],
              "max_depth":[3,5,8]}
              
gs_cv = GridSearchCV(lgbm, 
                     lgbm_params, 
                     cv = 10, 
                     n_jobs = -1, 
                     verbose = 2).fit(X, y)
gs_cv.best_params_

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X,y)

cross_val_score(lgbm_tuned, X, y, cv = 10).mean()

##89%

feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Significance Score Of Variables')

plt.ylabel('Variables')

plt.title("Variable Severity Levels")

plt.show()

![image](https://user-images.githubusercontent.com/100121721/166848775-dadbdf34-403e-43f8-9235-c1eb1082a17d.png)

