# data-science-portfolio
All Data Science Projects

#Import all the required libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
%matplotlib inline

from sklearn import metrics, model_selection, tree

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay, precision_score, roc_auc_score, accuracy_score, f1_score


import warnings
from warnings import filterwarnings
filterwarnings("ignore")


#Import the dataset and view the first five observations/rows
census = pd.read_csv("census.csv")
census.head()

![image](https://github.com/user-attachments/assets/7eee71d0-0859-458a-91c2-adf54bf52bdd)

#Column Information
census.shape
(45222, 14)
#There are a total of 45,222 rows and 14 columns out of which 9 are categorical and 5 are continuous.
#Out of the 9 categorical variables one is the outcome variable - income.

census.info()
![image](https://github.com/user-attachments/assets/da68356c-9d32-4e6e-88eb-185385fca6f4)

##Featureset Exploration
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

census.describe()

![image](https://github.com/user-attachments/assets/cfe18a8e-ced7-435f-8bd1-304aa5cd6ca0)


Correlation matrix
census.corr()
![image](https://github.com/user-attachments/assets/d244b691-04c7-45dc-8285-83e7af619dfe)


