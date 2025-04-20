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
- Understand the applicability and challenges associated with different datasets for the use of machine learning algorithms.

Correlation Heatmap:

![image](https://github.com/user-attachments/assets/bcc6c6d3-819a-460e-b26a-207463b49575)

Linear Regression:

![image](https://github.com/user-attachments/assets/603cf72f-30c6-48d3-b19c-ef8875e78e7b)


#### Unit 5 
Learned Objectives: 
- Clustering and relate that with algorithm logic.
- Algorithm logic

Watching the K-Means animation helped me grasp how the algorithm works beyond just the steps.

The first animation clearly demonstrated how K-Means clustering works step by step. It starts with randomly placing centroids, then repeatedly assigns points to the nearest centroid and updates the centroid positions based on the average location of those assigned points. I noticed that when centroids were placed too far from the main data distribution, they often ended up with very few or even no points assigned to them. In contrast, centroids that began closer to the centre of the data resulted in more balanced clusters and smoother convergence.

Convergence in K-Means occurs when the centroids no longer move significantly between iterations, meaning the cluster assignments have stabilised and further updates do not change the outcome. The animation made this process easy to visualise, as the centroids gradually shifted less and less with each step until they finally settled.

In the second animation, using the “Uniform Points” option, I manually selected the initial centroid positions. Regardless of the starting points, the algorithm consistently produced evenly spaced clusters. This showed that when data is uniformly distributed, K-Means tends to converge reliably, as there are no natural groupings that could mislead the algorithm.

Overall, the animations reinforced how important both initial centroid placement and data distribution are for K-Means to generate meaningful and consistent results and provided a clear view of how the algorithm reaches convergence during clustering. K-Means doesn’t consider the shape or density of clusters, it only focuses on distance to the centre. Hence, highlighting if the data isn’t roughly circular or evenly distributed, the results might not actually reflect the real structure. That’s something I’ll be more aware of when deciding whether or not K-Means is the right approach for a dataset.


Jaccard coefficient:

(Jack, Mary) = 3 / 7 = 0.429
(Jack, Jim) = 5 / 7 = 0.714
(Jim, Mary) = 1 / 7 = 0.143

#### Unit 7

Simple Perceptron:
```python
import numpy as np
inputs = np.array([45, 25])
# Check the type of the inputs
type(inputs)
# check the value at index position 0
inputs[0]
# creating the weights as Numpy array
weights = np.array([0.7, 0.1])
# Check the value at index 0 
weights[0]
def sum_func(inputs, weights):
    return inputs.dot(weights)
# for weights = [0.7, 0.1]
s_prob1 = sum_func(inputs, weights)
s_prob1
def step_function(sum_func):
  if (sum_func >= 1):
    print(f'The Sum Function is greater than or equal to 1')
    return 1
  else:
        print(f'The Sum Function is NOT greater')
        return 0
step_function(s_prob1 )
```
Output: 

![image](https://github.com/user-attachments/assets/811b753e-7e20-4003-8c17-e9be09bd8820)


#### Unit 8
Gradient Cost Function:
The following code is after changing the number of iteration and the learning rate and it is observed that the cost increases per iteration when the learning rate is increased.

```python
import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10     
    n = len(x)
    learning_rate = 0.08   

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)
```


Output:

![image](https://github.com/user-attachments/assets/396dce85-3dca-402e-a2fe-117eeb7177d7)


#### Unit 11 

Model Performance Measurement
Code:

```python
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
(tn, fp, fn, tp)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
SVC(random_state=0)
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()

plt.show()
```
Output:

![image](https://github.com/user-attachments/assets/7d1f3e9c-1bad-48fb-b41e-740aea9d1dce)


```python
from sklearn.metrics import f1_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

print(f"Macro f1 score: {f1_score(y_true, y_pred, average='macro')}")

print(f"Micro F1: {f1_score(y_true, y_pred, average='micro')}")

print(f"Weighted Average F1: {f1_score(y_true, y_pred, average='weighted')}")

print(f"F1 No Average: {f1_score(y_true, y_pred, average=None)}")

y_true = [0, 0, 0, 0, 0, 0]
y_pred = [0, 0, 0, 0, 0, 0]
f1_score(y_true, y_pred, zero_division=1)

# multilabel classification
y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
print(f"F1 No Average: {f1_score(y_true, y_pred, average=None)}")
```
Output:

![image](https://github.com/user-attachments/assets/7c20bccc-9e13-446c-8a06-a1483fbbe378)


```python
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
```
Output:

![image](https://github.com/user-attachments/assets/2d5a5060-c91b-4945-a224-fbb716aea219)


```python
from sklearn.metrics import precision_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
precision_score(y_true, y_pred, average='macro')
```
Output:

![image](https://github.com/user-attachments/assets/9174a1a6-7f41-4947-b334-d241c425a9db)


```python
from sklearn.metrics import recall_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
recall_score(y_true, y_pred, average='macro')
```
Output:

![image](https://github.com/user-attachments/assets/5c7d2219-0529-4c3a-999f-414c953811e4)


```python
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
```
Output:

![image](https://github.com/user-attachments/assets/38618dc7-896b-4905-af66-79fdb98c3c9c)


```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
roc_auc_score(y, clf.predict_proba(X)[:, 1])
```
Output:

![image](https://github.com/user-attachments/assets/097dacb6-63be-4be8-bf40-ea6736168727)


```python
#multiclass case
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(solver="liblinear").fit(X, y)
roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
```
Output:

![image](https://github.com/user-attachments/assets/9bf00bce-971f-4f5f-8b0d-92ba6deed525)


```python
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(
    svm.SVC(kernel="linear", probability=True, random_state=random_state)
)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
```

```python
plt.figure()
lw = 2
plt.plot(
    fpr[2],
    tpr[2],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
```
Output:

![image](https://github.com/user-attachments/assets/1c0b709e-4e75-424a-a1b3-d7f6f1581ee5)


```python
from sklearn.metrics import log_loss
log_loss(["spam", "ham", "ham", "spam"], [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
```
Output:

![image](https://github.com/user-attachments/assets/57870a4f-3214-4039-ac61-5c91446346c0)


```python
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)
```
Output (RMSE):

![image](https://github.com/user-attachments/assets/3e15d777-0e3f-4030-b7c0-e20aa9a2bc4a)


```python
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)
```
Output (MAE):

![image](https://github.com/user-attachments/assets/dc329d40-47a9-4260-9973-033bc84e2ac3)


```python
from sklearn.metrics import r2_score

r2_score(y_true, y_pred)
```
Output (r squared): 

![image](https://github.com/user-attachments/assets/e577cade-0699-4b00-86b2-af6bc083748b)

