# Simple_Linear_Regression 
## Objective
My aim is to predict house prices of new areas of houses based on the data I have.
## Data Collection 

I have a dataset which has 5 rows and 2 columns;
- 1st column consists the areas.
- 2nd column consists the prices of the houses.
## EDA 
First, load the dataset to be used into jupyter notebook;
```python
import pandas as pd
df = pd.read_csv(r'')
df.head()
```
## EDA 
Confirm the shape of dataset;
```python
df.shape
```
Descriptive statistics such as mean, median, max,min;
```python
df.describe ()
```
### Visualization
Plot the area against price;
```python
import matplotlib.pyplot as plt
plt.xlabel('Area', fontsize=20)
plt.ylabel('Price (USD), fontsize=20)
plt.scatter(df['area'], df['price'], marker='+', color='red')
plt. plot(df['area'], reg.predict(df[['area']]), color = 'blue') # added after model development
```

[!Output]()

## Model Development 
```python
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(df[['area']],df['price']) #train the model
```
