#!/usr/bin/env python
# coding: utf-8

# In[53]:


#Libraries reduired for this task
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


#importing and displaying data
df = pd.read_csv("E:\Python Practice\Prediction of House Sale\kc_house_data.csv\kc_house_data.csv")


# In[105]:


#Module 1: Importing Data Sets

df.head()
print(df.dtypes)


# In[56]:


df.describe()


# In[57]:


#Module 2: Data Wrangling

missing_data = df.isnull()


# In[58]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print()


# In[59]:


#data Warngling
df.drop(["id"], axis=1, inplace = True)


# In[60]:


df.dropna()


# In[61]:


df.describe()


# In[62]:


#Module 3: Exploratory Data Analysis

df['floors'].value_counts()


# In[63]:


df['floors'].value_counts().to_frame()


# In[64]:


sns.boxplot(x="waterfront", y="price", data=df)


# In[65]:


sns.regplot(data=df, x="sqft_above", y="price")
plt.ylim(0,)


# In[66]:


#Module 4: Model Development

lm = LinearRegression()
lm


# In[67]:


X = df[['sqft_living']]
Y = df['price']

lm.fit(X,Y)

lm.score(X,Y)


# In[68]:


y_data = df["price"]
x_data = df.drop("price", axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state = 1)

print("number of test samples:", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

lre = LinearRegression()

lre.fit(x_train[["sqft_living"]], y_train)
lre.score(x_test[["sqft_living"]], y_test)


# In[69]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]


# In[70]:


lm = LinearRegression()
lm

X = df[["floors"]]
Y = df["price"]

lm.fit(X, Y)
lm.score(X, Y)


# In[71]:


lm = LinearRegression()
lm

X = df[["waterfront"]]
Y = df["price"]

lm.fit(X, Y)
lm.score(X, Y)


# In[72]:


lm = LinearRegression()
lm

X = df[['lat']]
Y = df['price']

lm.fit(X,Y)
lm.score(X,Y)


# In[73]:


lm = LinearRegression()
lm

X = df[['bedrooms']]
Y = df['price']

lm.fit(X,Y)
lm.score(X,Y)


# In[74]:


lm = LinearRegression()
lm

X = df[['sqft_basement']]
Y = df['price']

lm.fit(X,Y)
lm.score(X,Y)


# In[75]:


lm = LinearRegression()
lm

X = df[['view']]
Y = df['price']

lm.fit(X,Y)

lm.score(X,Y)


# In[76]:


lm = LinearRegression()
lm

X = df[['bathrooms']]
Y = df['price']

lm.fit(X,Y)

lm.score(X,Y)


# In[77]:


lm = LinearRegression()
lm

X = df[['sqft_living15']]
Y = df['price']

lm.fit(X,Y)

lm.score(X,Y)


# In[78]:


lm = LinearRegression()
lm

X = df[['sqft_above']]
Y = df['price']

lm.fit(X,Y)

lm.score(X,Y)


# In[79]:


lm = LinearRegression()
lm

X = df[['grade']]
Y = df['price']

lm.fit(X,Y)

lm.score(X,Y)


# In[80]:


lm = LinearRegression()
lm

X = df[['sqft_living']]
Y = df['price']

lm.fit(X,Y)
lm.score(X,Y)


# In[81]:


Input = [("scale", StandardScaler()), ("polynominal", PolynomialFeatures(include_bias=False)), ("model", LinearRegression())]


# In[82]:


pipe = Pipeline(Input)
pipe


# In[83]:


pipe.fit(X, Y)


# In[84]:


pipe.score(X, Y)


# In[85]:


#Module 5: Model Evaluation and Refinement

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[86]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]

X=df[features]
Y=df["price"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[87]:


from sklearn.linear_model import Ridge


# In[88]:


pr = PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])
x_test_pr=pr.fit_transform(x_test[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[89]:


RidgeModel = Ridge(alpha=0.1)

RidgeModel.fit(x_train_pr, y_train)


# In[90]:


RidgeModel.score(x_train_pr, y_train)


# In[91]:


import tqdm as tqdm
from tqdm import tqdm
Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# In[93]:


width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


# In[94]:


from sklearn.preprocessing import PolynomialFeatures


# In[95]:


pr = PolynomialFeatures(degree=2)
pr


# In[96]:


x_train_pr=pr.fit_transform(x_train[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[97]:


x_polly=pr.fit_transform(x_train[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[98]:


RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train_pr, y_train)
RidgeModel.score(x_train_pr, y_train)


# In[100]:


x_test_pr=pr.fit_transform(x_test[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])
x_polly=pr.fit_transform(x_test[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])

RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_test_pr, y_test)
RidgeModel.score(x_test_pr, y_test)


# In[ ]:




