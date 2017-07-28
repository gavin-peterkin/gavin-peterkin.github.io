
# A Quick Try at Titanic Kaggle Competition

## Intro

Kaggle has hosted the [titanic competition](https://www.kaggle.com/c/titanic) for a few years now. It gives aspiring data scientists the opportunity to test some basic classification approaches on a real dataset that's familiar to almost anyone.

## Motivation

My main goal with this attempt is to see how well I can do in a short amount of time. I'm not going to spend more than a day (8 hrs) on this in part becuase I think it seems kind of silly to try to eek out each bit of predictive power from the data when this is really just meant to be an interesting practice exercise.


```python
from __future__ import division, print_function

# Basic imports. I'm using python 2

import numpy as np
import pandas as pd

import re

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV, train_test_split, cross_val_score
)
from sklearn.ensemble import ( 
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)

%matplotlib inline
```


```python
train = pd.read_csv('data/train.csv', index_col=0)
```


```python
train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(test.shape)
print(train.shape)
```

    (418, 10)
    (891, 11)



```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 1 to 891
    Data columns (total 11 columns):
    Survived    891 non-null int64
    Pclass      891 non-null int64
    Name        891 non-null object
    Sex         891 non-null object
    Age         714 non-null float64
    SibSp       891 non-null int64
    Parch       891 non-null int64
    Ticket      891 non-null object
    Fare        891 non-null float64
    Cabin       204 non-null object
    Embarked    889 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 83.5+ KB


# Feature Engineering

This seems like one of the most important and often overlooked parts of the modeling process. Unfortunately, as part of my effort to get respectable results as quickly as possible, I'm not going to spend as much time working on it as I should.

Rather than messing with my modeling approach, I could probably significantly improve my score by just making/using _better_, more predictive, features.


```python
columns_to_investigate = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

fig, axes = plt.subplots(
    len(columns_to_investigate), 1, figsize=(8, 20)
)

survived_mask = (train['Survived'] == 1)

for ax, col in zip(axes, columns_to_investigate):
    ax.hist(
        train.loc[survived_mask, col].fillna(-1), bins=30,
        normed=True, color='g', label='Survived', alpha=0.4
    )
    ax.hist(
        train.loc[~survived_mask, col].fillna(-1), bins=30,
        normed=True, color='r', label='Died', alpha=0.4
    )
    ax.set_title(col)
    ax.legend()
    
fig.tight_layout()
```


![png](output_7_0.png)



```python
train['Fare'].describe()
```




    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64




```python
title_re = re.compile(r' ([A-Za-z]+)\.')
title_replacements = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}

cabin_letter_re = re.compile(r'([A-Z]+)')
cabin_num_re = re.compile(r'(\d+)')

# TODO: Ensure the cuts are the same!

def add_title_name_len(df):
    """
    Adds Title col astype int from Name
    Drops the name
    """
    df['Title'] = df['Name'].str.extract(title_re)
    df['Title'] = df['Title'].replace([
        'Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major',
        'Rev', 'Sir', 'Jonkheer', 'Dona'
    ], 'Rare')
    df['Title'] = (
        df['Title']
        .map(title_replacements)
        .map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
        .fillna(0)
        .astype(int)
    )
    df['NameLen'] = df['Name'].str.len()
    df.drop('Name', axis=1, inplace=True)

def clean_sex(df):
    """self-explanatory"""
    df['Sex'] = df['Sex'].fillna(df['Sex'].mode())
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

def add_family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
def classify_ages(df):
    """
    Adds MissingAge column
    Makes Age an int column with categories from 1 to 5 ordered by age
    """
    age_cuts = [15, 20, 30, 40, 50, 60]
    df['MissingAge'] = 0.
    df.loc[(df['Age'].isnull()), 'MissingAge'] = 1.
    df['Age'] = pd.cut(
        df['Age'], bins=age_cuts, labels=range(1, len(age_cuts))
    ).astype(float)
    df['Age'] = df['Age'].fillna(0).astype(int)

def clean_fare(df):
    """
    Adds a MissingFare column
    Fills missing fares with 0s
    Adds a quantile for fare
    """
    cuts = [4, 7, 10, 20, 50, 100, 200]
    # Don't expect this to be missing
    df['Fare'] = df['Fare'].fillna(0)
    df['QuantFare'] = pd.cut(
        df['Fare'], cuts, labels=range(1, len(cuts))
    ).astype(float)

def clean_embarked(df):
    """
    Fill the very few missing with mode
    Maps to integers
    """
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
    df['Embarked'] = (
        df['Embarked']
        .fillna(df['Embarked'].mode()[0])
        .map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    )

def clean_cabin(df):
    """
    Adds MissingCabin column
    Adds a CabinLetter column with letters mapped to ints
    Adds a CabinNumber column of type ints
    Drops original Cabin column
    """
    df['MissingCabin'] = 0
    df.loc[df['Cabin'].isnull(), 'MissingCabin'] = 1
    df.loc[df['Cabin'].isnull(), 'Cabin'] = ''
    df['CabinLetter'] = (
        df['Cabin'].str.extract(cabin_letter_re).fillna('')
    )
    df['CabinNumber'] = (
        df['Cabin'].str.extract(cabin_num_re)
        .astype(float)
        .fillna(0)
        .astype(int)
    )
    df.drop('Cabin', axis=1, inplace=True)

def categorize(df, columns):
    """
    Given a df and list of columns, converts those columns to dummies
    concats them with the df and drops the original columns.
    """
    for col in columns:
        dummy_df = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df.drop(col, axis=1, inplace=True)
        df = pd.concat([df, dummy_df], axis=1)
    return df
    
def data_prep(df):
    add_title_name_len(df)
    clean_sex(df)
    classify_ages(df)
    add_family_size(df)
    df.drop('Ticket', axis=1, inplace=True)
    clean_fare(df)
    # On second thought, try dropping this. There's not reason to think it's predictive
#     clean_embarked(df)
    df.drop('Embarked', axis=1, inplace=True)
    clean_cabin(df)
    return categorize(df, ['Title', 'Age', 'QuantFare', 'CabinLetter'])
```


```python
train = pd.read_csv('data/train.csv', index_col=0)
train = data_prep(train)
```

    /Users/gavinpeterkin/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:14: FutureWarning: currently extract(expand=None) means expand=False (return Index/Series/DataFrame) but in a future version of pandas this will be changed to expand=True (return DataFrame)
      
    /Users/gavinpeterkin/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:86: FutureWarning: currently extract(expand=None) means expand=False (return Index/Series/DataFrame) but in a future version of pandas this will be changed to expand=True (return DataFrame)
    /Users/gavinpeterkin/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:89: FutureWarning: currently extract(expand=None) means expand=False (return Index/Series/DataFrame) but in a future version of pandas this will be changed to expand=True (return DataFrame)



```python
train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>NameLen</th>
      <th>MissingAge</th>
      <th>FamilySize</th>
      <th>MissingCabin</th>
      <th>CabinNumber</th>
      <th>Title_2</th>
      <th>Title_3</th>
      <th>Age_1</th>
      <th>Age_2</th>
      <th>Age_3</th>
      <th>Age_4</th>
      <th>Age_5</th>
      <th>QuantFare_2.0</th>
      <th>QuantFare_3.0</th>
      <th>QuantFare_4.0</th>
      <th>QuantFare_5.0</th>
      <th>QuantFare_6.0</th>
      <th>CabinLetter_A</th>
      <th>CabinLetter_B</th>
      <th>CabinLetter_C</th>
      <th>CabinLetter_D</th>
      <th>CabinLetter_E</th>
      <th>CabinLetter_F</th>
      <th>CabinLetter_G</th>
      <th>CabinLetter_T</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>23</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>51</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>85</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>22</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>44</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>123</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>24</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 10));

sns.heatmap(train.iloc[:, :11].corr(), square=True, annot=True);
```


![png](output_12_0.png)


# Investigating Feature Usefulness


```python
mini_df = train.iloc[:, :10].copy()

columns_to_investigate = mini_df.columns[1:10]

fig, axes = plt.subplots(9, 1, figsize=(10, 30))

survived_mask = (mini_df['Survived'] == 1)

for i, (col, ax) in enumerate(zip(columns_to_investigate, axes)):
#     print(i, col, ax)
    ax.hist(
        mini_df.loc[survived_mask, col], bins=30,
        normed=True, color='g', label='Survived', alpha=0.4
    )
    ax.hist(
        mini_df.loc[~survived_mask, col], bins=30,
        normed=True, color='r', label='Died', alpha=0.4
    )
    ax.set_title(col)
    ax.legend()
fig.tight_layout()
```


![png](output_14_0.png)


One thing of note: missing data seems to almost always be a bad sign. It's probably correlated with wealth in some way since the lower class "less important" individuals are probably less likely to have complete information available.


```python
y_train = train.pop('Survived').values
X_train = train.values
```

# Parameter selection via grid search CV

I'm going to try out the following models:
* RandomForestClassifier
* AdaBoostClassifier
* BaggingClassifier
* GradientBoostingClassifier
* SupportVectorClassifier

Of course by fitting all of these hyperparameters to the training data, I run the risk of starting to _meta overfit_ the training data by overtuning the hyperparameters to just the training data. That's not something I'm particularly worried about with this project though.

Generally in cases where there is a `n_estimators` parameter, I use a low learning rate with a larger number of estimators or iterations to limit the risk of overfitting.


```python
# A running dictionary of all my models, so I can keep track

all_ensembles = dict()
```

## Random Forest


```python
rfc = RandomForestClassifier()

rfc_param_grid = {
    'n_estimators': [500],
    'max_features': np.arange(3, 7, 1, dtype=int),
    'min_samples_split': [4, 5, 6]
}

rfc_g = GridSearchCV(
    rfc, rfc_param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1
)
```


```python
rfc_g.fit(X_train, y_train)
```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   31.2s
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:   42.7s finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'n_estimators': [500], 'max_features': array([3, 4, 5, 6]), 'min_samples_split': [4, 5, 6]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='accuracy', verbose=1)




```python
print(rfc_g.best_score_)
# 0.826

# May be able to get better accuracy with larger number of estimators

print(rfc_g.best_params_)
# {'max_features': 6, 'min_samples_split': 6, 'n_estimators': 500}
```

    0.810325476992
    {'max_features': 3, 'min_samples_split': 6, 'n_estimators': 500}



```python
rf_model = RandomForestClassifier(
    n_estimators=3000, max_features=3, min_samples_split=6
)

rf_scores = (
    cross_val_score(
        rf_model, X_train, y_train, scoring='accuracy', verbose=1, n_jobs=-1
    )
)
```

    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:   11.3s finished



```python
rf_score = rf_scores.mean()
print(rf_score)
# 0.828

rf_model.fit(X_train, y_train)
```

    0.799102132435





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features=3, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=6, min_weight_fraction_leaf=0.0,
                n_estimators=3000, n_jobs=1, oob_score=False,
                random_state=None, verbose=0, warm_start=False)




```python
all_ensembles.update({'rf': (rf_model, rf_score)})
```

## AdaBoost Classifier


```python
abc = AdaBoostClassifier()

abc_param_grid = {
    'n_estimators': [1000],
    'learning_rate': [0.01, 0.1, 0.3]
}

abc_g = GridSearchCV(
    abc, abc_param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1
)
```


```python
abc_g.fit(X_train, y_train)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits


    [Parallel(n_jobs=-1)]: Done  15 out of  15 | elapsed:   12.8s finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=None),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'n_estimators': [1000], 'learning_rate': [0.01, 0.1, 0.3]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='accuracy', verbose=1)




```python
print(abc_g.best_score_)
# 0.801346

print(abc_g.best_params_)
# {'learning_rate': 0.1, 'n_estimators': 1000}
```

    0.791245791246
    {'n_estimators': 1000, 'learning_rate': 0.3}



```python
abc_model = AdaBoostClassifier(
    n_estimators=2000, learning_rate=0.1
)

# Past 1500 estimators doesn't help as much here

abc_scores = (
    cross_val_score(
        abc_model, X_train, y_train, scoring='accuracy', verbose=1, n_jobs=-1
    )
)
```

    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:    5.4s finished



```python
abc_score = abc_scores.mean()
print(abc_score)
# 0.795

abc_model.fit(X_train, y_train)
```

    0.793490460157





    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=0.1, n_estimators=2000, random_state=None)




```python
all_ensembles.update({'abc': (abc_model, abc_score)})
```

## Bagging


```python
bag = BaggingClassifier()

bag_param_grid = {
    'base_estimator': [None],
    'n_estimators': [10, 25, 30, 40],
    'max_samples': [0.2, 0.3, 0.5, 0.7, 1.],
    'max_features': [0.95, 1.]
}

bag_param = GridSearchCV(
    bag, bag_param_grid, scoring='accuracy', n_jobs=-1, cv=5,
    verbose=1
)
```


```python
bag_param.fit(X_train, y_train)
```

    Fitting 5 folds for each of 40 candidates, totalling 200 fits


    [Parallel(n_jobs=-1)]: Done 144 tasks      | elapsed:    4.2s
    [Parallel(n_jobs=-1)]: Done 200 out of 200 | elapsed:    6.1s finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=BaggingClassifier(base_estimator=None, bootstrap=True,
             bootstrap_features=False, max_features=1.0, max_samples=1.0,
             n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
             verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'n_estimators': [10, 25, 30, 40], 'max_samples': [0.2, 0.3, 0.5, 0.7, 1.0], 'base_estimator': [None], 'max_features': [0.95, 1.0]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='accuracy', verbose=1)




```python
print(bag_param.best_score_)
# 0.8226

print(bag_param.best_params_)
# {'base_estimator': None,
#  'max_features': 1.0,
#  'max_samples': 0.3,
#  'n_estimators': 25}
```

    0.809203142536
    {'max_features': 0.95, 'max_samples': 0.5, 'base_estimator': None, 'n_estimators': 30}



```python
bag_model = BaggingClassifier(
    n_estimators=25, max_features=1., max_samples=0.3
)

bag_scores = (
    cross_val_score(
        bag_model, X_train, y_train, scoring='accuracy', verbose=1, cv=5
    )
)
```

    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.3s finished



```python
bag_score = bag_scores.mean()
print(bag_score)
# 0.802

bag_model.fit(X_train, y_train)
```

    0.80922488117





    BaggingClassifier(base_estimator=None, bootstrap=True,
             bootstrap_features=False, max_features=1.0, max_samples=0.3,
             n_estimators=25, n_jobs=1, oob_score=False, random_state=None,
             verbose=0, warm_start=False)




```python
all_ensembles.update({'bagging': (bag_model, bag_score)})
```

# GradientBoosting


```python
gb_classifier = GradientBoostingClassifier()

gb_param_grid = {
    'learning_rate': [0.1],
    'n_estimators': [3000],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.75, 1]
}

gb_param = GridSearchCV(
    gb_classifier, gb_param_grid, scoring='accuracy', n_jobs=-1, cv=5,
    verbose=1
)
```


```python
gb_param.fit(X_train, y_train)
```

    Fitting 5 folds for each of 9 candidates, totalling 45 fits


    [Parallel(n_jobs=-1)]: Done  45 out of  45 | elapsed:  1.1min finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=1,
                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                  n_estimators=100, presort='auto', random_state=None,
                  subsample=1.0, verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'n_estimators': [3000], 'subsample': [0.5, 0.75, 1], 'learning_rate': [0.1], 'max_depth': [3, 5, 7]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='accuracy', verbose=1)




```python
print(gb_param.best_score_)
# 0.82267

print(gb_param.best_params_)
# {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 3000, 'subsample': 0.5}
```

    0.781144781145
    {'n_estimators': 3000, 'subsample': 1, 'learning_rate': 0.1, 'max_depth': 7}



```python
gb_model = GradientBoostingClassifier(
    learning_rate=0.1, max_depth=7, n_estimators=3000, subsample=0.95
)
```


```python
gb_scores = (
    cross_val_score(
        gb_model, X_train, y_train, scoring='accuracy', verbose=1, cv=5, n_jobs=-1
    )
)
```

    [Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:   11.3s finished



```python
gb_score = gb_scores.mean()
print(gb_score)
# 0.80

gb_model.fit(X_train, y_train)
```

    0.769955106538





    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=7,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=1,
                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                  n_estimators=3000, presort='auto', random_state=None,
                  subsample=0.95, verbose=0, warm_start=False)




```python
all_ensembles.update({'gb': (gb_model, gb_score)})
```

# Support Vector Machine Classifier


```python
svc = SVC()

svc_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('svc', svc)
])


svc_grid_params = [{
        'svc__kernel': ['poly'],
        'svc__degree': np.arange(1, 8, dtype=int),
        'svc__C': np.logspace(-2, 3, 8)
    },
    {
        'svc__kernel': ['rbf'],
        'svc__gamma': np.logspace(-4, 2, 8),
        'svc__C': np.logspace(-2, 2, 10)
    }
]

svc_grid = GridSearchCV(
    svc_pipe, param_grid=svc_grid_params, n_jobs=-1,
    verbose=2
)
```


```python
svc_grid.fit(X_train, y_train)
```

    Fitting 3 folds for each of 136 candidates, totalling 408 fits
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.01 ....................
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.01 ....................
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.01 ....................
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=1, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=1, svc__kernel=poly, svc__C=0.01, total=   0.1s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=1, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=2, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=2, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] ..... svc__degree=3, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.01 ....................
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=2, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=3, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] ..... svc__degree=3, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.01 ....................
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=5, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=6, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=7, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] ..... svc__degree=5, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] ..... svc__degree=4, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.01 ....................
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=6, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=5, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.01 ....................
    [CV] ..... svc__degree=4, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.01 ....................
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] ..... svc__degree=7, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.01 ....................
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] ..... svc__degree=4, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] ..... svc__degree=6, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] ..... svc__degree=7, svc__kernel=poly, svc__C=0.01, total=   0.0s
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=0.0517947467923, total=   0.1s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=0.0517947467923, total=   0.1s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.0517947467923 .........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=0.0517947467923, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV] svc__degree=1, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV] svc__degree=5, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=0.268269579528, total=   0.1s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=0.268269579528, total=   0.1s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV] svc__degree=7, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV] svc__degree=3, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV] svc__degree=6, svc__kernel=poly, svc__C=0.268269579528 ..........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV] svc__degree=5, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=0.268269579528, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV] svc__degree=5, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV] svc__degree=4, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV] svc__degree=5, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=1.38949549437 ...........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=1.38949549437, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=1.38949549437, total=   0.1s
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV] svc__degree=5, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=7.19685673001, total=   0.1s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV] svc__degree=7, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=7.19685673001 ...........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=7.19685673001, total=   0.1s
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=7.19685673001, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV] svc__degree=4, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV] svc__degree=4, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV] svc__degree=3, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV] svc__degree=4, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV] svc__degree=7, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=37.2759372031 ...........


    [Parallel(n_jobs=-1)]: Done 108 tasks      | elapsed:    2.3s


    [CV]  svc__degree=3, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=37.2759372031, total=   0.1s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=37.2759372031 ...........
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=37.2759372031, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV] svc__degree=3, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=193.069772888, total=   0.2s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV] svc__degree=4, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=193.069772888, total=   0.2s
    [CV]  svc__degree=3, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV] svc__degree=1, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV]  svc__degree=4, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV] svc__degree=5, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=193.069772888, total=   0.2s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=5, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV]  svc__degree=6, svc__kernel=poly, svc__C=193.069772888, total=   0.1s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=1000.0 ..................
    [CV]  svc__degree=1, svc__kernel=poly, svc__C=193.069772888, total=   0.4s
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=193.069772888 ...........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=1000.0 ..................
    [CV]  svc__degree=7, svc__kernel=poly, svc__C=193.069772888, total=   0.0s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=1000.0 ..................
    [CV]  svc__degree=2, svc__kernel=poly, svc__C=193.069772888, total=   0.4s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=4, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=4, svc__kernel=poly, svc__C=1000.0, total=   0.0s
    [CV] svc__degree=4, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=4, svc__kernel=poly, svc__C=1000.0, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=5, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=5, svc__kernel=poly, svc__C=1000.0, total=   0.0s
    [CV] svc__degree=5, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=5, svc__kernel=poly, svc__C=1000.0, total=   0.0s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=2, svc__kernel=poly, svc__C=1000.0, total=   0.7s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=1, svc__kernel=poly, svc__C=1000.0, total=   0.6s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=6, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=6, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=3, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=6, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] ... svc__degree=6, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.01 .................
    [CV] ... svc__degree=3, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=3, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] .. svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.01 .................
    [CV] ... svc__degree=7, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] .. svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.01 .................
    [CV] ... svc__degree=3, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.01 ......
    [CV] ... svc__degree=7, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__degree=7, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] .. svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.01, total=   0.1s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.01 ......
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.01 ......
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.01 .......
    [CV] ... svc__degree=1, svc__kernel=poly, svc__C=1000.0, total=   1.2s
    [CV] svc__degree=1, svc__kernel=poly, svc__C=1000.0 ..................
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.01 .......
    [CV] ... svc__degree=7, svc__kernel=poly, svc__C=1000.0, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.01 .........
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.01 ........
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.01, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.01 .......
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.01 .........
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.01 ........
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.01 ..........
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.01 .........
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.01 ........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] ... svc__degree=1, svc__kernel=poly, svc__C=1000.0, total=   0.2s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.01 ..........
    [CV] svc__degree=2, svc__kernel=poly, svc__C=1000.0 ..................
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.01 ..........
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.01 ..........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.01 ..........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0278255940221 ......
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.01 ..................
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.01, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.01 ..........
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0278255940221 ......
    [CV] ... svc__gamma=100.0, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.01 ..................
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0278255940221 ......
    [CV] ... svc__gamma=100.0, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.01 ..................
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] ... svc__gamma=100.0, svc__kernel=rbf, svc__C=0.01, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV] ... svc__degree=2, svc__kernel=poly, svc__C=1000.0, total=   0.5s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__degree=2, svc__kernel=poly, svc__C=1000.0 ..................
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.1s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.1s
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0774263682681 ......
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0278255940221 .......
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0278255940221 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0774263682681 ......
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0278255940221 .......
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0774263682681 ......
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0278255940221 .......
    [CV] ... svc__degree=2, svc__kernel=poly, svc__C=1000.0, total=   0.5s
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0278255940221, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.1s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.1s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.1s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.215443469003 .......
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0774263682681 .......
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0774263682681 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.215443469003 .......
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0774263682681 .......
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.1s
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.215443469003 
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.215443469003 .......
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0774263682681 .......
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.0774263682681, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.215443469003, total=   0.1s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.215443469003 
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.215443469003 
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.599484250319 .......
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.215443469003 ........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.215443469003, total=   0.1s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.599484250319 .......
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.215443469003 ........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.215443469003, total=   0.1s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.215443469003, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.215443469003 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.599484250319 .......
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.215443469003 ........
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.215443469003, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.599484250319 
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.599484250319 
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.599484250319 
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.599484250319 
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.599484250319 
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.599484250319 ........
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=1.6681005372 .........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=1.6681005372 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.599484250319 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=1.6681005372 .........
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.599484250319 ........
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=1.6681005372 
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=1.6681005372 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.599484250319, total=   0.1s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=0.599484250319 ........
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=1.6681005372 .........
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=1.6681005372 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=1.6681005372 
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=1.6681005372 
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=0.599484250319, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=1.6681005372 .
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=1.6681005372 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=1.6681005372 
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=1.6681005372 ..
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=1.6681005372 
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=1.6681005372 .
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=1.6681005372 ..
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=4.64158883361 ........
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=1.6681005372 ..
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=1.6681005372 .
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=1.6681005372 ..........
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=1.6681005372 ..
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=4.64158883361 ........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=1.6681005372 ..
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=1.6681005372 ..........
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=4.64158883361 ........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=1.6681005372 ..
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=1.6681005372 ..........
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=4.64158883361 
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=1.6681005372, total=   0.1s
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=1.6681005372, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=4.64158883361 
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=4.64158883361 .
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=4.64158883361, total=   0.1s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=4.64158883361 .
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=4.64158883361 .
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=4.64158883361, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=4.64158883361 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=4.64158883361, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=4.64158883361 .
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=12.9154966501 ........
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=4.64158883361 .........
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=4.64158883361 .
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=12.9154966501 ........
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=4.64158883361 .........
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=4.64158883361 .
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=12.9154966501 ........
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=4.64158883361 .........
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=4.64158883361, total=   0.0s
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=12.9154966501 
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=4.64158883361, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=12.9154966501 
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=12.9154966501 .
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=12.9154966501 
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=12.9154966501 .
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=12.9154966501 .
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=12.9154966501 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=12.9154966501 .........
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=35.938136638 .........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=12.9154966501 .
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=35.938136638 .........
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=12.9154966501 .
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=12.9154966501 .........
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=12.9154966501 .
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=12.9154966501, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=12.9154966501 .........
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=35.938136638 .........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=35.938136638 
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV]  svc__gamma=0.0001, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=35.938136638 
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=35.938136638 
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=12.9154966501, total=   0.1s
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=35.938136638 
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=35.938136638 .
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=35.938136638 ..
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=35.938136638 
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=35.938136638 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=35.938136638 
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=35.938136638 .
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=35.938136638 
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=35.938136638 
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=35.938136638 ..
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=35.938136638 .
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=35.938136638 ..
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=100.0 ................
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=35.938136638 ..
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=35.938136638 ..........
    [CV] . svc__gamma=0.0001, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=100.0 ................
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=35.938136638 ..
    [CV] . svc__gamma=0.0001, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=35.938136638 ..
    [CV] svc__gamma=0.0001, svc__kernel=rbf, svc__C=100.0 ................
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=35.938136638 ..........
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] . svc__gamma=0.0001, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=35.938136638, total=   0.0s
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=100.0 .....
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=35.938136638 ..........
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=100.0 .....
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=100.0 ......
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=100.0 ........
    [CV] svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=100.0 .....
    [CV]  svc__gamma=100.0, svc__kernel=rbf, svc__C=35.938136638, total=   0.1s
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=100.0 .......
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=100.0 .........
    [CV]  svc__gamma=0.000719685673001, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=100.0 ......
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=100.0 ........
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=100.0 .......
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=100.0 ......
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=100.0 .........
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=100.0 .......
    [CV]  svc__gamma=0.00517947467923, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=100.0 .........
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=100.0 ........
    [CV]  svc__gamma=0.0372759372031, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=100.0 .................
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=100.0 .........
    [CV] .. svc__gamma=100.0, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=100.0 .................
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=100.0 .........
    [CV]  svc__gamma=0.268269579528, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=100.0 .........
    [CV] .. svc__gamma=100.0, svc__kernel=rbf, svc__C=100.0, total=   0.0s
    [CV] svc__gamma=100.0, svc__kernel=rbf, svc__C=100.0 .................
    [CV]  svc__gamma=13.8949549437, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV]  svc__gamma=1.93069772888, svc__kernel=rbf, svc__C=100.0, total=   0.1s
    [CV] .. svc__gamma=100.0, svc__kernel=rbf, svc__C=100.0, total=   0.0s


    [Parallel(n_jobs=-1)]: Done 408 out of 408 | elapsed:   10.0s finished





    GridSearchCV(cv=None, error_score='raise',
           estimator=Pipeline(steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svc', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))]),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid=[{'svc__degree': array([1, 2, 3, 4, 5, 6, 7]), 'svc__kernel': ['poly'], 'svc__C': array([  1.00000e-02,   5.17947e-02,   2.68270e-01,   1.38950e+00,
             7.19686e+00,   3.72759e+01,   1.93070e+02,   1.00000e+03])}, {'svc__gamma': array([  1.00000e-04,   7.19686e-04,   5.17947e-03,   3...   5.99484e-01,   1.66810e+00,   4.64159e+00,   1.29155e+01,
             3.59381e+01,   1.00000e+02])}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=2)




```python
print(svc_grid.best_score_)
# 0.818

print(svc_grid.best_params_)
# {'svc__C': 1.6681005372000592,
#  'svc__gamma': 0.26826957952797248,
#  'svc__kernel': 'rbf'}
```

    0.801346801347
    {'svc__gamma': 0.037275937203149381, 'svc__kernel': 'rbf', 'svc__C': 4.6415888336127775}



```python
svc_model = SVC(
    C=30., kernel='rbf', gamma=0.005
)

svc_scores = (
    cross_val_score(
        svc_model, X_train, y_train, scoring='accuracy', verbose=1, cv=5
    )
)

svc_score = svc_scores.mean()
print(svc_score)
# 0.773
```

    0.763226300426


    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.2s finished



```python
svc_model.fit(X_train, y_train)

all_ensembles.update({'svc': (svc_model, svc_score)})
```

# XGBoost


```python
import xgboost as xgb
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train)
```

    /Users/gavinpeterkin/anaconda2/lib/python2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
def add_empty_missing_cols(df, all_columns):
    for col in all_columns:
        if col not in df.columns and col != 'Survived':
            df[col] = 0.
    return df

# Preparing the test data
test = pd.read_csv('data/test.csv', index_col=0)
test = data_prep(test)
test = add_empty_missing_cols(test, train.columns)

X_test = test.values
```

    /Users/gavinpeterkin/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:14: FutureWarning: currently extract(expand=None) means expand=False (return Index/Series/DataFrame) but in a future version of pandas this will be changed to expand=True (return DataFrame)
      
    /Users/gavinpeterkin/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:86: FutureWarning: currently extract(expand=None) means expand=False (return Index/Series/DataFrame) but in a future version of pandas this will be changed to expand=True (return DataFrame)
    /Users/gavinpeterkin/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:89: FutureWarning: currently extract(expand=None) means expand=False (return Index/Series/DataFrame) but in a future version of pandas this will be changed to expand=True (return DataFrame)



```python
xgb_model = xgb.XGBClassifier(
    max_depth=3, n_estimators=3000, learning_rate=0.005
)

```


```python
xgb_score = cross_val_score(xgb_model, X_train, y_train).mean()
print(xgb_score)
# On mini test from within training
# 0.8226
```

    0.801346801347



```python
all_ensembles.update({'xgb': (xgb_model, xgb_score)})
```


```python
xgb_model.fit(X_train, y_train)

y_test_pred = xgb_model.predict(X_test)

final_results = pd.DataFrame(
    {'PassengerId': test.index.values, 'Survived': y_test_pred}
)

final_results.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_results.to_csv('data/final_result.csv', index=False)

# Got a 76
```

# Ensembling


```python
def ultra_ensembled_model(fit_estimators, X):
    """
    This isn't really serious, but it will be interesting to see how
    it does.
    """
    score_normalization = np.sum([v[1] for v in fit_estimators.values()])
    predictions = np.zeros((X.shape[0], len(fit_estimators)))
    for i, (name, (estimator, score)) in enumerate(fit_estimators.iteritems()):
        weight = score / score_normalization
        try:
            predictions[:, i] = estimator.predict(X) * weight
        except:
            print("ERROR:", name)
    return (predictions.sum(axis=1) > 0.5).astype(int)
        
```


```python
y_pred = ultra_ensembled_model(all_ensembles, X_train)

acc = (y_pred == y_train).sum() / y_pred.size

print("Accuracy:", acc)

# Accuracy: 0.950617283951
# This is on the training data, which is what it was fit to, so there's
# probably a lot of overfitting
```

    Accuracy: 0.909090909091



```python
y_test_pred = ultra_ensembled_model(all_ensembles, X_test)

final_results = pd.DataFrame(
    {'PassengerId': test.index.values, 'Survived': y_test_pred}
)
```


```python
final_results.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_results.to_csv('data/final_result.csv', index=False)
```

# Conclusion / Final Result

In the end, my best score was from the ensemble, where I got a score of 78%. While, I do think it's possible to improve that, it will most likely involve a lot of time sitting and thinking really carefully about feature engineering.

## How are people getting >90% accuracy?

I'm extremely skeptical of the submissions in which people get accuracies of anything greater than ~85-90%. It just doesn't seem remotely possible to be able to predict whether someone would live or die that night only on the basis of their income/class, gender, age, and name. There are just too many other factors at play. Any score that high was achieved by _other_ means (i.e. fitting the test data).
