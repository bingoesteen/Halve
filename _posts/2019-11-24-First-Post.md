---
layout: post
title: "Analyzing why people quit their job and who will quit next. Using Python and Machine Learning, Decision Tree Algorithm."
categories:
  - Post Formats
tags:
  - Data Analyst
  - Machine Learning
---

# Introduction

According to the article by Balance Articles, employees jump across jobs and roles for averagely 12 times during their lifetime career. Shocking! Imagine what happened to all of those billions of dollars wasted put by the Multinational Companies (MNCs) to retain the employees.

Therefore, managing employee attrition effectively and efficiently is very important. New employees will spend lots of time and money to train and hire them which could otherwise be reallocated in another investment. In fact, this has been a big concern for IBM HR Professionals that IBM is investing billions of dollars in Watson to predict flight risk and win employee attrition. Surely this means knowing why employee quits is a great beginning to attract and retain talents.

In conclusion,it is very important to derive a data driven decision making to understand employees leave to reduce turnover rate, save hiring/training cost and maximize work productivity. All of these translate into great profit for the years ahead.

I have done some analysis using Python and one of Machine Learning algorithm to find out why people leave their job and who the next potential employees that predicted will leave the job. Here we go! (Ba Dum Tss!)

# 1. Understanding the dataset

The dataset comes from my repo [here](https://github.com/bhaskoro-muthohar/DataScienceLearning/blob/master/HR_comma_sep.csv) with csv (coma seperated value) format. The metadata includes the following features: Employee satisfaction level, Last Evaluation, Number of Projects, Average Monthly Hours, Time Spent at the Company, Whether they have had a work accident, Whether they have had a promotion in the last 5 years, Departments, Salary, Whether the employee has left.

Let us start our journey by importing the csv file into Pandas Dataframe.

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('https://raw.githubusercontent.com/bhaskoro-muthohar/DataScienceLearning/master/HR_comma_sep.csv')

```

# 2. Viewing the data

Let us just glimpse on the quick and dirty sanity test at the data below with a simple code.
```python
print("This is Head")
print(df.head())


print('This is Tail')

print(df.tail())

```
![image](https://media.licdn.com/dms/image/C5112AQHEiyeca_S2Ow/article-inline_image-shrink_1000_1488/0?e=1580342400&v=beta&t=c1gTgNo-qVyvgU1mJOvrJ5x9jCzxnJdSWVQGxohsV0c)

First, to understand the quick outlook of our dataset, let us describe the dataset. We will transpose it to put the features as rows for better view.
```python
df.describe().T
```
![image](https://media.licdn.com/dms/image/C5112AQFIQ77xoVd6nA/article-inline_image-shrink_1000_1488/0?e=1580342400&v=beta&t=TZxaGyaCMJFiUxHH4Qb1dgUwXKUtTUmej47OH4s046g)

This will describe all of the numerical features with their aggregated values (counts, means, etc). From here we could see that everything looks good: full count values, no null, and logical distributed values. One interesting value is the max average monthly hours: 310 hours. Wow! That means somebody is working 15 hours per weekday. This guy is surely a workholic.

We then can describe the non numerical values by including the datatype ‘object’ for parameter. include=[‘object’]
```python
df.describe(include=['object'])
```
![image](https://media.licdn.com/dms/image/C5112AQHT18dsCtZstQ/article-inline_image-shrink_1000_1488/0?e=1580342400&v=beta&t=f4oc35eh3JT9EPqRQJ_xSKpqwEXNeLhMoR17HI4dnSE)

The sales here means the department that the employees are working on and the salary indicates “top”,”medium”,”low”. OK, all datas seems clean. We will proced.

# 3. Data Exploration

## Importing Matplotlib and Seaborn for Story Telling
```python
# Import seaborn and matplotlib with matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
```
After importing seaborn and matplotlib, let us get the counts of people that left or stayed. We will visualize it in a simple matplotlib pie chart.

Let us generate multiple plot distribution for each leaving and staying employees, then combine it with Matplotlib Figure. This code will be a bit long as we are inserting 10 subplots (5 rows, 2 columns). But the visualization is really valuable to discover the trend and discrepancy among those who left and stayed. Hence the story telling!
```python
#make seperated list for satisfaction level
x1 = list(leftdf['satisfaction_level'])
x2 = list(notleftdf['satisfaction_level'])


#make seperated list for last evaluation
x3 = list(leftdf['last_evaluation'])
x4 = list(notleftdf['last_evaluation'])


#make seperated list for number project
x5 = list(leftdf['number_project'])
x6 = list(notleftdf['number_project'])


#make seperated list for average monthly hours
x7 = list(leftdf['average_montly_hours'])
x8 = list(notleftdf['average_montly_hours'])


#make seperated list for time spent company
x9 = list(leftdf['time_spend_company'])
x10 = list(notleftdf['time_spend_company'])



#assign color and names
colors = ['#E69F00', '#56B4E9']
names = ['leaving', 'stay']


# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(511)
plt.hist([x1,x2], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('satisfaction level')
plt.ylabel('normalized value')


# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(512)
plt.hist([x3,x4], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('last evaluation')
plt.ylabel('normalized value')


# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(513)
plt.hist([x5,x6], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('number projects')
plt.ylabel('normalized value')


# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(514)
plt.hist([x7,x8], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('average monthly hours')
plt.ylabel('normalized value')


# Make the histogram using a list of lists
#Normalize and assign colors and names
plt.subplot(515)
plt.hist([x9,x10], bins=12, normed=True, color=colors, label=names)
#plot formatting
plt.legend()
plt.xlabel('time spent')
plt.ylabel('normalized value')


plt.subplots_adjust(top=11 ,bottom=10)
plt.tight_layout()

plt.show()
```
![image](https://media.licdn.com/dms/image/C5112AQHPdaG5a7rJGg/article-inline_image-shrink_1000_1488/0?e=1580342400&v=beta&t=rS-JZxQX9C9yr0XZv0XnS299ZikYzT2wjl1vGeXa1e8)

Now, what could we learn from this?

## Insights: Profile of the people who left

* Satisfaction_level = People who left tends to had low satisfaction_level, but there is some people who very satisfied with their job and still left the job.
* last_evaluation = High and low which might indicate over achiever and under achiever that leaves the companies. If this is true, that means employees left for two reasons: that they feel they could not channel their talents or motivation well or that they are motivated and apply for better career opportunities.
* Number_project = Most have 2 projects. Maybe unlike the distribution of the staying employees who have 3–4 projects. We might want to compare this further with the average_monthly_hours and viewed whether there is a Simpson Paradox ongoing where the insights changed as we consider other confound variables. We also need to compare further with the size and nature of the project.
* average_monthly_hours = It has either large or not so much average worked hours. This is unique as maybe the employees are getting too much or too less engaged within their company.
* time_spend_company = Some of them spend less time than the employees who stay, we might want to assume that they are less engaged with their workload. Combined with average_monthly_hours, this is a likely assumption.
* Work_accident Does not have much work accidents
* promotion_last_5years Lot less not promoted
* sales Does not differ much
* salary Most of them are at the lower level salary (low)

# 4. Correlation Analysis

```python
corr = leftdf.drop(‘left’,axis=1).corr()sns.heatmap(corr)
```
![image](https://media.licdn.com/dms/image/C5112AQFhx-jsSw4NsA/article-inline_image-shrink_1000_1488/0?e=1580342400&v=beta&t=ecKVn7-e_NuNFlj8VYbDWLfTSYIHA8HNPQKp6DeDgBI)

Seaborn saves the day! Very simple visualization, all in one line sns.heatmap().
## Insights: Correlated Elements
1. number_project and average_monthly_hours: this makes sense as the more projects you have, the more time you should spend on it. This could be possibly the reason of dissatisfaction.
2. last_evaluations and average_monthly_hours: This is somewhat a good and hopeful find, this shows that the longer the monthly hours, the more likely you get a good last evaluation. We need to figure out further background information about the data to extract some hypothesis from this.

# 5. Department Analysis
## Which departments whose people leave most often?
We will use Seaborn Factorplot to visualize left data frame based on the departments and salary levels. 

```python
sns.factorplot(x="sales",data=leftdf,col="salary",kind='count',aspect=1,size=2.5)
```
![image](https://media.licdn.com/dms/image/C5112AQF3f6sX4d-4Pw/article-inline_image-shrink_1500_2232/0?e=1580342400&v=beta&t=YxKAaTB6sr0y41VaFr2dQzLSd4GZp522sMmOKCEW2m8)
## Insights
1. Seems like Sales left the companies most of the time in both low and medium salaries.
2. Similarly, it was also followed by technical and support for both low and medium salaries.

We could now ask ourselves *“why?”*

## Why do employees from these departments left?
Let us visualize the boxplot to get clear answers why. Using Seaborn we could stack the visualizations inside the matplotlib figure. This will allow us to align the four features which we had covered earlier.
```python
# plot each pie chart in a separate subplot
plt.subplot(411)
sns.boxplot(x='sales', y='satisfaction_level', data=leftdf)


# plot each pie chart in a separate subplot
plt.subplot(412)
sns.boxplot(x='sales', y='time_spend_company', data=leftdf)


# plot each pie chart in a separate subplot
plt.subplot(413)
sns.boxplot(x='sales', y='number_project', data=leftdf)


# plot each pie chart in a separate subplot
plt.subplot(414)
sns.boxplot(x='sales', y='average_montly_hours', data=leftdf)


plt.tight_layout()

plt.show()
```
![image](https://media.licdn.com/dms/image/C5112AQHs8lPW8HwAQg/article-inline_image-shrink_1500_2232/0?e=1580342400&v=beta&t=dJ-xehQ4gZgBrVzgobKZwxmcd81P1J7GwdprURg6nvA)

## Insights:
1. In terms of the comparisons with other departments whose employees left. Sales left no significant remarks why they left compared to other departments. In fact, we could see here that their surveyed satisfaction level is actually higher than accounting and other departments. This means that probably the only reason why sales left are because of low salary. Another reason is that they manipulated the survey answers probably due to fear of being found out by their sales managers. This could be a blackmailing material in case they do not meet the sales target.

2. Accounting has one of the lowest satisfaction level with most of the quartiles located below 0.5. However, at the same time, accounting is one of the department where the rate of leaving is low.

3. Technical and support, on the other hand, gave a wide range from all features examined. Unlike sales, we could still see large number of employees who are not satisfied enough with their work. Thus, we need to isolate this further to know why some of them left. Probably, small companies and big companies will treat these employees differently. 

4. Marketing and product management have high satisfaction levels despite high rate of quitting in low salary. This might be because this industry moves really fast and some of the practitioners took better opportunities at multi national companies.

# Model Generation
tet teret teret! Now we will move to Machine Learning Section. Hooray!!
## Creating train test split
Let us start by creating the train and test data set. The idea is to train the model using using the training data set and test the model using the test data set. This is to avoid overfitting the model to reach high accuracy but high variance error. Another set that you could use is the validation set which is to tune the model.
![image](https://media.licdn.com/dms/image/C5112AQHnZvXID4oAJQ/article-inline_image-shrink_1500_2232/0?e=1580342400&v=beta&t=AFftQQMlBVOGdXzE6wHm85x4X5sqVX5IQJsEHezJywI)

```python
def process_df_for_ml(df):
    """
    Process a dataframe for model training/prediction use.


    Returns X/y tensors.
    """


    df = df.copy()
    # Map salary to 0,1,2
    df.salary = df.salary.map({"low": 0, "medium": 1, "high": 2})
    # dropping left and sales X for the df, y for the left
    X = df.drop(["left", "sales"], axis=1)
    y = df["left"]
import numpy as np
from sklearn.model_selection import train_test_split


X, y = process_df_for_ml(df)


#splitting the train and test sets

X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0, stratify=y)    return (X, y)
```
Once we have done the split, we will proceed to train and validate the model.

## Training Decision Tree

Decision tree is basically a binary tree flowchart where each node splits a group of observations according to some feature variable. The goal of a decision tree is to split your data into groups such that every element in one group belongs to the same category. This will be based on the lowest mean squared to make sure that each group has the homogeneity defined. The biggest advantage of this is that it is really intuitive and easy to use to derive insights.

Please refer the following article for more information. https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

Let us start training our decision tree.

```python
from sklearn import tree
# Train a decision tree.
X, y = process_df_for_ml(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
clftree = tree.DecisionTreeClassifier(max_depth=3)

clftree.fit(X_train, y_train)
```
The biggest perk of having decision tree is to visualize it. You can import pydotplus and run the following visualizations.
![image](https://media.licdn.com/dms/image/C5112AQHkMuWmfeVUXg/article-inline_image-shrink_1500_2232/0?e=1580342400&v=beta&t=ruC-qbeJhCpo4YCSbeY7I1hIm38K4LAvzi44HoCxIA0)

## Insights: What most likely cause the people to leave?
1. low satisfaction level (<=0.115)
2. high number of projects (>2.5).
3. Low to medium last evaluation (≤0.575)
4. Salaries are not much of an important predictor in this level

## Decision Tree Explained
In general, we get satisfaction level as our top level node. Then it branches out to level 2 nodes with the number_project and time_spend_company. Technically the lower the node position, the less gini (classification dispersion) of the branch node is. This means that the lower the node, the more homogeneous the separations become.

The top node holds the most important feature to separate the classifications, followed by the second level and so on.

## Conclusion: 3 simple suggestions for companies to retain employees
1. Reduce the number of projects into 1 or 2 for the employees at any given time. This is more important than reducing their working hours.
2. Salaries are not much of an important predictor except for Sales, Technical, and Support. Exercise Salary Raise with cautions.
3. Improve the communication and trust between managers and employees. A big portion of the employees who left received very low last evaluation score. This means we need to foster healthy relationships among the departments/groups. One of the concrete solutions is to reduce micromanagers and penalties of errors made.

## Who will leave next?
Before make a prediction for who will leave next. I want to show you my model score is.
```python
print("Test set score: {:.2f}".format(clftree.score(X_test, y_test)))
```
Test set score: 0.96

It means that my model can predict with accuracy 96%! Awesome.

Now we will predict the people who will leave next.

```python
# Test the decision tree on people who haven't left yet.
notleftdf = df[df["left"] == 0].copy()

X, y = process_df_for_ml(notleftdf)

# Plug in a new column with ones and zeroes from the prediction.

notleftdf["will_leave"] = clftree.predict(X)

# Print those with the will-leave flag on.

print(notleftdf[notleftdf["will_leave"] == 1])
```
![image](https://media.licdn.com/dms/image/C5112AQHT_mrpXwobog/article-inline_image-shrink_1500_2232/0?e=1580342400&v=beta&t=QhwlvbLCyDuFTg9Sb3W27KUW2JLX9jEEaG0BXrvRGpE)
There are 424 potential employees predicted will leave the company.

Thank you so much for reading. For your information, I am right now on the path to became Junior Data Scientist and this work can't happen without help of many people from stackoverflow, reddit.com/r/MLQuestion, towardsdatascientist, etc.
