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

#Datatypes of each column of the data
census.dtypes

![image](https://github.com/user-attachments/assets/752837ab-43f4-49e1-9c8e-c4310636cbf1)

Names of all the columns along with their count of distinct values
print(census.nunique())

![image](https://github.com/user-attachments/assets/f5bc7ed2-1f47-4fcc-a003-52311cc05c38)

Checking for null values
census.isnull().sum()

![image](https://github.com/user-attachments/assets/b26044a9-5f81-4d17-aff5-40d1127df84c)

Therefore, our dataset doesn't contain any null values

Distribution
census[['age','capital-gain','capital-loss','hours-per-week']].hist(figsize=(10,8));

![image](https://github.com/user-attachments/assets/41cf19e0-7140-4c09-add1-6e8a97671a53)


We see the distribution of columns age, catpital-gain, capital-loss, and hours-per-week above. We see that all of them are skewed to the right

WORKCLASS
workclass= census['workclass'].value_counts()

sns.set() # setting the background of the figure

plt.figure(figsize=(15,10),dpi=140) # we are plotting the figure by specifying the figure size and resolution

c = plt.cm.cool # Linear segmented color map

plt.title("Workclass", fontsize=18) # title for the graph
plt.xlabel('Count', fontsize=12) # we are setting the label for x-axis  using specified fontsize 
plt.ylabel('Workclass Type', fontsize=12) # we are setting the label for y-axis  using specified fontsize 

sns.barplot(workclass.values, workclass.index,
            palette=[c(0.1),c(0.2),c(0.3),c(0.4),c(0.5),c(0.6),c(0.7),c(0.8),c(0.9),c(0.99)])

for i, v in enumerate(workclass.values): # Enumerated object
    plt.text(0.8,i,v,color='k',fontsize=14) #plotting  text for each level

   ![image](https://github.com/user-attachments/assets/4953119b-1432-4d19-95d3-6aabba22fc88)

The working class is highly unbalanced with majority working in Private

EDUCATION
education = census.groupby('education_level')['education-num'].value_counts()
print(education)

import os
print(os.linesep)

top_5_education = census.groupby('education_level')['education-num'].value_counts().sort_values(ascending=False).head(5)
print(top_5_education)

![image](https://github.com/user-attachments/assets/edcbfed4-31fe-4ce3-b61f-897c1399b67c)

sns.set() # setting the background of the figure

plt.figure(figsize = (8,6), dpi =105) # we are plotting the figure by specifying the figure size and resolution

plt.title('Top five Education Level',fontsize=18) # title for the graph

colors = ['r','b','c','y','m'] # setting the color for each portion of the piechart

plt.pie(top_5_education.values,labels=top_5_education.index,colors = colors,autopct = '%d') 
# plotting piechart for the top five directors with respective color for their portion

plt.show() # Display the figure

![image](https://github.com/user-attachments/assets/20604a15-2e25-496b-a553-03228dd31cf0)

We see that majority are High School Graduates, and the capacity decreases as the education level increases

COUNTRIES
top_countries = census['native-country'].value_counts().head(10)
top_countries

![image](https://github.com/user-attachments/assets/037d134f-4230-452b-8ab1-618ab912c004)

Most of the observations were taken from people working in the United States

MARTIAL STATUS
marital_status = census['marital-status'].value_counts()

sns.set() # setting the background of the figure

plt.figure(figsize=(15,10),dpi=140) # we are plotting the figure by specifying the figure size and resolution

c = plt.cm.cool # Linear segmented color map

plt.title("Marital Status", fontsize=18) # title for the graph
plt.xlabel('Count', fontsize=12) # we are setting the label for x-axis  using specified  fontsize 
plt.ylabel('Marital status type', fontsize=12) # we are setting the label for y-axis  using specified  fontsize 

sns.barplot(marital_status.values, marital_status.index,
            palette=[c(0.1),c(0.2),c(0.3),c(0.4),c(0.5),c(0.6),c(0.7),c(0.8),c(0.9),c(0.99)])
# barplot of top 10 martial status with palette (colors to use for the different levels of the graph)

for i, v in enumerate(marital_status.values): # Enumerated object
    plt.text(0.8,i,v,color='k',fontsize=14) #plotting  text for each level

  ![image](https://github.com/user-attachments/assets/975d0d0c-6377-4997-8556-118689f41d88)

Most of the observations are from people who are married

RACE
race = census['race'].value_counts()
print(race)

sns.set() # setting the background of the figure

plt.figure(figsize = (8,6), dpi =105) # we are plotting the figure by specifying the figure size and resolution

plt.title('Race',fontsize=18) # title for the graph

colors = ['cyan','violet','yellow','green','skyblue'] # colors for each portio of the donut plot

my_circle = plt.Circle((0,0),0.7,color ='white') #create a circle with radius 0.7

plt.pie(race.values, labels=race.index, colors=colors,autopct = '%1.1f%%' ) 
# plotting donut plot with respective colors and their repective % (portion) to the data.

p=plt.gcf() # Get the current figure.

p.gca().add_artist(my_circle) # get the axes

plt.show() # Display the figure

![image](https://github.com/user-attachments/assets/3fc59d50-c12a-4a9e-8dca-c33988cbf5e6)

Again, the distribution for Race is highly unbalanced with 86% of the population being predominantly White/Caucasian.

GENDER
gender = census['sex'].value_counts()
print(gender)

sns.set() # setting the background of the figure

plt.figure(figsize = (8,6), dpi =105) # we are plotting the figure by specifying the figure size and resolution

plt.title('Gender',fontsize=18) # title for the graph

colors = ['grey','cyan'] # colors for each portio of the donut plot

my_circle = plt.Circle((0,0),0.7,color ='white') #create a circle with radius 0.7

plt.pie(gender.values, labels=gender.index, colors=colors,autopct = '%1.1f%%' )
# plotting donut plot with respective colors and their repective % (portion) to the data.

p=plt.gcf() # Get the current figure.

p.gca().add_artist(my_circle) # get the axes

plt.show() # Display the figure

![image](https://github.com/user-attachments/assets/35b7950e-800e-468d-8d34-87881deb03cc)

Gender is slighty unbalanced with 67.5% males and 32.5% Females

HOURS-PER-WEEK
hours = census['hours-per-week'].value_counts().head(10)
sns.set() # setting the background of the figure

plt.figure(figsize=(15,10),dpi=140) # we are plotting the figure by specifying the figure size and resolution

c = plt.cm.cool  # Linear segmented color map

plt.title("Hours-per-week", fontsize=18) # title for the graph
plt.xlabel('Working hours per week', fontsize=12) # we are setting the label for x-axis  using specified  fontsize 
plt.ylabel( 'count', fontsize=12) # we are setting the label for y-axis  using specified  fontsize 

sns.barplot(hours.index, hours.values,
            palette=[c(0.1),c(0.2),c(0.3),c(0.4),c(0.5),c(0.6),c(0.7),c(0.8),c(0.9),c(0.99)])
# barplot of top 10 genres with palette (colors to use for the different levels of the graph)

for i, v in enumerate(hours.values): # Enumerated object
    plt.text(0.8,i,v,color='k',fontsize=14) #plotting  text for each level

  ![image](https://github.com/user-attachments/assets/1656dd4f-2024-4ee4-bad4-105a5583e223)

It is clear from the barchart above that majority of the working class works for about 40 hours a week

Preparing the Data
cvdata = KFold(n_splits=5, random_state=None, shuffle=False) # Cross Validation

income_raw = census['income']
features = census.drop(columns = "income")
Scaling the Data
# Initialize Standard scaler, then apply it to the features
scaler = StandardScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_transform = pd.DataFrame(data = features)
features_transform[numerical] = scaler.fit_transform(features_transform[numerical])
features_transform.head()

![image](https://github.com/user-attachments/assets/03f8affb-55b5-4ce9-b6e9-5789bd1d98f2)

Data Preprocessing
# One-hot encode the 'features_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_transform)

# Encode the 'income_raw' data to numerical values
income = income_raw.map({'<=50K':0, '>50K':1})

features_final.head()
![image](https://github.com/user-attachments/assets/03c7f4fa-3b36-45e9-9f89-6b24b0e22876)


Shuffle and Split Data
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size = 0.33, random_state = 42, 
                                                    stratify = income)
Evaluating Model Performance
1. KNN
2. Logistic Regression
3a. Single Tree
3b. Bagged Tree
3c. Random Forest

KNN
Rsquared = []
for k in range(1,51):   
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, features_final, income, cv=cvdata)
    Rsquared.append(scores.mean())
plt.plot(range(1,51),Rsquared)

![image](https://github.com/user-attachments/assets/d19fc789-3f44-42dd-bd57-6a203788b04d)

max(Rsquared)

#finding the best k value
range(1,51)[np.argmax(Rsquared)]
33
knn_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors = 33))
scores = cross_val_score(knn_pipeline, features_final, income, cv=cvdata)
scores.mean()
0.3721383787334939

![image](https://github.com/user-attachments/assets/a40cbc5c-5c63-4793-b6f9-d9c97819e610)

