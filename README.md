# E - Portfolio
## About me 
I am a Masters student pursuing Data science based in the UAE.


### Skills 
- Data Anlysis
- Business analytics
- Data Visualization
- Machine Learning Algorithms
- Deep Learning

### Expertise 
Python, R, Google Analytics 4, Tableau, power BI, Jira, SQLplus , MySQL, SAP 

## Projects
Here is a summary of all the details of my Machine Learning project so far!

### Machine Learning [Jan 2025]
#### Introduction 
In this unit, I delved into the evolving landscape of machine learning and its significance in deriving insights from complex data. Through hands-on experience and theoretical exploration, I developed an understanding of how algorithms can be trained to recognize patterns, make predictions, and support data-driven strategies. Machine learning emerged not just as a technical skillset, but as a practical tool for solving real-world problems transforming raw information into actionable intelligence across diverse business contexts.

#### Expected Learning outcmes: 
- Understand the fundamentals of machine learning concepts and algorithms
- Learn how to apply machine learning techniques to real-world datasets
- Develop the ability to critically evaluate model performance and outcomes
- Engage in self-reflection on the ethical and practical implications of machine learning

#### How I’ve applied these concepts in practice

The activities paired with self reflection really made me aware of where I stood in terms of my personal development.

### Unit 1-2 
Learned objectives:
- Focused on understanding the fundamentals of idioms and grammer by participating in a collaborative discussion. 
- The comparison done by my peers hekped me expand my learning to see different perspectives.

### Unit 3 
Learned Objectives : 
- Understanding how how the change in data points impacts correlation and regression.

#### Pearson's correlation 
The following results are what was observed when we changed the variable values.

##### No Noise
Code:

![no noise code](https://github.com/user-attachments/assets/b80760e7-79c5-45b0-baa7-3f6d04250740)

Output:

![no noise output](https://github.com/user-attachments/assets/efff2288-cd23-4c7a-bf41-fb3380cad8fb)


When we eliminate the noise, we are looking at a perfect one to one relationship in which every point falls neatly into place, forming a straight line which makes the Pearson correlation hit the max 1.0000.

##### Less Noise 

Code:

![small noise code](https://github.com/user-attachments/assets/ddee93cf-1a1d-414d-9d9e-94e07b668ea9)

Output:

![small noise output](https://github.com/user-attachments/assets/14ed5904-e4eb-4887-8384-40c18241b420)

Adding a bit of noise, that perfect line will start to wobble slightly. The connection is still very strong and the correlation stays high around 0.9 to 0.99.

##### High Noise 

Code:

![high noise code](https://github.com/user-attachments/assets/5ba60a13-5f94-4fdb-a392-c0e521882eef)

Output:

![high noise](https://github.com/user-attachments/assets/81ca37ce-9bce-46fd-93df-3aea87e6c83a)

When throwing in a lot of noise, things start to fall apart. The link between the two variables weakens, the correlation drops and the scatterplot starts to look like a cloud than a line. It is harder to spot any clear trend.


#### Linear Regression
When looking at linear regression, the way your data behaves really makes all the difference. If there is a clear trend and the numbers don’t vary too much, the line fits nicely and the connection between variables is easy to see. Even when adding a bit of noise the pattern still holds, but it is just a little messier. The more randomness you throw in, the harder it becomes to spot any solid link—the data spreads out and the line doesn’t really capture what is going on. Outliers, even just one or two weird values, can throw everything off and totally shift the direction of the line. And if there’s no actual relationship between the variables, you will notice the line pretty much flattens out, because there is nothing meaningful to predict.

Code:

![ex2 code](https://github.com/user-attachments/assets/f049cf36-3a9d-41ef-8b87-da4164b66865)

Output:

![ex2 chart](https://github.com/user-attachments/assets/45131851-0047-4f64-94e2-12809c96e015)

##### Predict Future Values

If you tweak the data, the whole pattern shifts, so the line the model draws will change too. With more consistent values, predictions usually improve. But if we try to predict way beyond the original data range, It becomes less reliable at that stage because the model is essentially estimating without solid reference points.

![predicting future values ex2](https://github.com/user-attachments/assets/0191cc84-8400-4308-a640-260ff46f6683)

#### Multiple Linear Regression

Code: 
```python
import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
```
Output:
![co2 predicted](https://github.com/user-attachments/assets/bc26c666-1cd0-4bee-b28f-d84f8735ca59)

```python
print(regr.coef_)
```
Output:
![coefficient](https://github.com/user-attachments/assets/84dd9496-da3e-4137-b502-f4f2c88b5e9c)

```python
predictedCO2 = regr.predict([[3300, 1300]])

print(predictedCO2)
```
Output:
![co2 predicted](https://github.com/user-attachments/assets/f17bffa3-e15f-4538-b45a-6eaab18eaf1f)

#### Unit 5 
Learned Objectives: 
