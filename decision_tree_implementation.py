import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score,classification_report

# Reading dataset using pandas
df=pd.read_csv('Dataset.csv')
df.head()

# Converting categorical columns into numerical values
label_encoders={}
for col in df.select_dtypes(include=['object']).columns:
  le=LabelEncoder()
  df[col]=le.fit_transform(df[col])
  label_encoders[col]=le

#Selecting X and Y features from dataset
x=df.drop(columns=['credit_risk'])
y=df['credit_risk']

# Splitting the dataset into training and testing subsets
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

# 1.Decision tree using Entorpy
# model fitting and calculating accuracy of the model using entropy
model_entropy=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=42)
model_entropy.fit(x_train,y_train)

# prediciting the model
y_pred_entropy=model_entropy.predict(x_test)

# evaluating accuracy_score and classifcation_report
accuracy_entropy=accuracy_score(y_test,y_pred_entropy)
print(f"Model accuracy (entropy): {accuracy_entropy:2f}")
print("Classification Report: ",classification_report(y_test,y_pred_entropy))

# plotting the tree for visualization using entropy
plt.figure(figsize=(15,8))
plot_tree(model_entropy,feature_names=x.columns,class_names=['GOOD','BAD'],filled=True,rounded=True)
plt.title("Decision Tree Visualization(entropy)")
plt.show()

# 2.Decision tree using Gini impurity
model_gini=DecisionTreeClassifier(criterion='gini',max_depth=5,random_state=42)
model_gini.fit(x_train,y_train)

# predicting the model
y_pred_gini=model_gini.predict(x_test)

# evaluation of accuracy score and calssification report
accuracy_gini=accuracy_score(y_test,y_pred_gini)
print(f"Accuracy Score(gini) is: {accuracy_gini:2f}")
print("Classification report is: ",classification_report(y_test,y_pred_gini))

# plotting the figure based on gini impurity
plt.figure(figsize=(15,8))
plot_tree(model_gini,feature_names=x.columns,class_names=['GOOD','BAD'],filled=True,rounded=True)
plt.title("Decision Tree Visualization(gini)")
plt.show()
