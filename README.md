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

Pearson's correlation 
The following results are what was observed when we changed the variable values.

No Noise
Code:

![no noise code](https://github.com/user-attachments/assets/b80760e7-79c5-45b0-baa7-3f6d04250740)

Output:

![no noise output](https://github.com/user-attachments/assets/efff2288-cd23-4c7a-bf41-fb3380cad8fb)


When we eliminate the noise, we are looking at a perfect one to one relationship in which every point falls neatly into place, forming a straight line which makes the Pearson correlation hit the max 1.0000.

Less Noise 

Code:

![small noise code](https://github.com/user-attachments/assets/ddee93cf-1a1d-414d-9d9e-94e07b668ea9)

Output:

![small noise output](https://github.com/user-attachments/assets/14ed5904-e4eb-4887-8384-40c18241b420)

Adding a bit of noise, that perfect line will start to wobble slightly. The connection is still very strong and the correlation stays high around 0.9 to 0.99.

High Noise 

Code:

![high noise code](https://github.com/user-attachments/assets/5ba60a13-5f94-4fdb-a392-c0e521882eef)

Output:

![high noise](https://github.com/user-attachments/assets/81ca37ce-9bce-46fd-93df-3aea87e6c83a)

When throwing in a lot of noise, things start to fall apart. The link between the two variables weakens, the correlation drops and the scatterplot starts to look like a cloud than a line. It is harder to spot any clear trend.


Linear Regression
When looking at linear regression, the way your data behaves really makes all the difference. If there is a clear trend and the numbers don’t vary too much, the line fits nicely and the connection between variables is easy to see. Even when adding a bit of noise the pattern still holds, but it is just a little messier. The more randomness you throw in, the harder it becomes to spot any solid link—the data spreads out and the line doesn’t really capture what is going on. Outliers, even just one or two weird values, can throw everything off and totally shift the direction of the line. And if there’s no actual relationship between the variables, you will notice the line pretty much flattens out, because there is nothing meaningful to predict.

Code:

![ex2 code](https://github.com/user-attachments/assets/f049cf36-3a9d-41ef-8b87-da4164b66865)

Output:

![ex2 chart](https://github.com/user-attachments/assets/45131851-0047-4f64-94e2-12809c96e015)

Predict Future Values

If you tweak the data, the whole pattern shifts, so the line the model draws will change too. With more consistent values, predictions usually improve. But if we try to predict way beyond the original data range, It becomes less reliable at that stage because the model is essentially estimating without solid reference points.

![predicting future values ex2](https://github.com/user-attachments/assets/0191cc84-8400-4308-a640-260ff46f6683)

Multiple Linear Regression

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

Polynomial Regression
```python
import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

#NumPy has a method that lets us make a polynomial model
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

#specify how the line will display, we start at position 1, and end at position 22
myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
```
Output:

![ex4 1](https://github.com/user-attachments/assets/6d17b786-8254-45a0-9522-20bc047193c4)

```python
import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))
```
Output: 

![ex4 2](https://github.com/user-attachments/assets/f5d971a8-b3aa-4870-a6ef-9fa606ecf820)

```python
import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

speed = mymodel(17)
print(speed)
```
Output:

![ex4 3](https://github.com/user-attachments/assets/f70e8ddc-fee3-407d-b4a5-7799c0ce3ffd)

#### Unit 4
Learned Objectives: 
- Apply and critically appraise machine learning techniques to real-world problems, particularly where technical risk and uncertainty is involved.

EDA Tutorial
Missing Values Identified:
- horsepower has 6 missing values.

Here’s what we found in the numerical columns:

Skewness:
```output
mpg             0.457092
cylinders       0.508109
displacement    0.701669
horsepower      1.087326
weight          0.519586
acceleration    0.291587
model year      0.019688
origin          0.915185
```
Highest skewness:

- horsepower

- origin

- displacement

Lowest skewness (almost symmetric):

- model year


Kurtosis:
```output
mpg            -0.515993
cylinders      -1.398199
displacement   -0.778317
horsepower      0.696947
weight         -0.809259
acceleration    0.444234
model year     -1.167446
origin         -0.841885
```
Most peaked (high kurtosis):

- horsepower

- acceleration

Light tails (platykurtic):

- cylinders

- model year

Highest skewness:

horsepower: 1.09 (right-skewed)

- origin

- displacement

Lowest skewness (almost symmetric):

- model year

#### Unit 5 
Learned Objectives: 
- Clustering and relate that with algorithm logic.
- Algorithm logic

Watching the K-Means animation helped me grasp how the algorithm works beyond just the steps.

The first animation clearly demonstrated how K-Means clustering works step by step. It starts with randomly placing centroids, then repeatedly assigns points to the nearest centroid and updates the centroid positions based on the average location of those assigned points. I noticed that when centroids were placed too far from the main data distribution, they often ended up with very few or even no points assigned to them. In contrast, centroids that began closer to the centre of the data resulted in more balanced clusters and smoother convergence.

Convergence in K-Means occurs when the centroids no longer move significantly between iterations, meaning the cluster assignments have stabilised and further updates do not change the outcome. The animation made this process easy to visualise, as the centroids gradually shifted less and less with each step until they finally settled.

In the second animation, using the “Uniform Points” option, I manually selected the initial centroid positions. Regardless of the starting points, the algorithm consistently produced evenly spaced clusters. This showed that when data is uniformly distributed, K-Means tends to converge reliably, as there are no natural groupings that could mislead the algorithm.

Overall, the animations reinforced how important both initial centroid placement and data distribution are for K-Means to generate meaningful and consistent results and provided a clear view of how the algorithm reaches convergence during clustering. K-Means doesn’t consider the shape or density of clusters, it only focuses on distance to the centre. Hence, highlighting if the data isn’t roughly circular or evenly distributed, the results might not actually reflect the real structure. That’s something I’ll be more aware of when deciding whether or not K-Means is the right approach for a dataset.
