# Simple Linear Regression 
## Objective
My aim is to predict house prices of new areas of houses based on the data I have.
## Data Collection 

I have a [dataset](Assets/house_price.csv) in csv format which has 5 rows and 2 columns;
- 1st column consists the areas.
- 2nd column consists the prices of the houses.
## EDA (Exploratory Data Analysis)
First, load the dataset to be used into jupyter notebook;
```python
import pandas as pd
df = pd.read_csv(r'file_path.csv')
df.head()
```

Confirm the shape of dataset;
```python
df.shape
```
Descriptive statistics such as mean, median, max, min;
```python
df.describe()
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

![Output](Images/Scatter_plot.PNG)

## Model Development 
```python
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(df[['area']],df['price']) #train the model
```

Predict prices to check accuracy of the model;
```python
import numpy as np
areas = df[['area']]

def predict_for_areas(start_area, end_area):
areas = np.arange(start_area, end_area+1).reshape(-1,1)

df['predicted price'] = reg.predict(areas)
df
```

Compute Mean Squared Error;
```python
from sklearn.metrics import mean_squared_error
actual = df['price']
pred = df['predicted price']
MSE = mean_squared_error(actual,pred)
MSE
```

## Model interpretation
- Having a Root Mean Squared Error(RMSE) of 
```python
RMSE = mean_squared_error(actual,pred,squared= False)
RMSE
```
**13668.030893042014**, this means the average error in predictions is about $13,668.

Since the prices are in hundreds of thousands, an RMSE of about $13,668 is acceptable.

Concludes that the model is applicable.

y = mx+b 
- y = represents the price of the house(dependent)
- m = coefficient
- x = represents the area of the house(independent) 
- b = intercept

Compute the intercept and coefficient;
```python
reg.intercept_
```

```python
reg.coef_
```

- Having a positive _coefficient_ of **135.78767123**, means that as the area of house increase by one square feet, the predicted price of house increases approximately by $135.79

- Having an _intercept_ of **180616.43835616432**, means that when the area of the house is 0, the predicted price of house is approximately $180616.44

  - In practical terms, this might not make sense, as a house with an area of zero square feet does not exist. However, the intercept is crucial for the mathematical formulation of the regression line.
