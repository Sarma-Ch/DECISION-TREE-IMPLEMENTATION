COMPANY: CODTECH IT SOLUTIONS

NAME: CHAVALI NARASIMHA SARMA

INTERN ID: CT06DM1358

DOMAIN: MACHINE LEARNING

DURATION: 6 WEEKS

MENTOR: NEELA SANTHOSH KUMAR

DESCRIPTION:

A Decision Tree is a Supervised Machine Learning Algorithm used for classification and regression tasks. It models decisions and their possible cmsequences as a tree like structure made up of nodes and branches.

A Decision Tree Classifier splits data into subsets based on the value od input features. Each internal node represents o decison or a feature, each branch represents the outcome of the decison and each leaf node represents a final class label.

Decision Trees are used when the data is clearly seperable and it is easy to explain. Feature interactions are non_linear or complex.

Types of Decision Trees:

Classification Trees: Used when the target variable is categorical. These trees predict a class label.

Regression Trees: Used when the target variable is continous. They predict a numerical value.

These Decision Trees are manily used in:

1.Medical diagnosis (To predict disease based on symptoms). 2.Credit scoring (To classify good or bad credit risk). 3.Marketing (Segmenting customers for targeted promotions). 4.Fraud detection (Detecting anamolies in transactions). 5.Spam email detection (Filtering spam or non_spam based on some words in the mail).

Advantages of Decision Trees: 1.Easy to interpret and visualize.

2.Handles both numerical and categorical data.

3.No need for standardization and normalization of data.

4.Works well if the data is missing in the data set.

Limitations of Decision Trees: 1.Prone to overfitting

2.Sensitivity to noisy data

3.Less powerful on it's own comapred to ensemble methods like RandomForests or GradientBoostedTrees.

There are mainly two measures used for the decision tree classifier:

Gini Impurity
Entropy
Gini Impurity:

It is used in CART(Classification Regression Trees). Gini measures how often a randomly chosen sample would be incorrectly calssified if it were labeled randomly based on class distribution. The formula used to calculate is : [Gini=1-sigma (1 to n)*Probability of class(Pi)] if Gini=0 the node is perfectly purity. Gini is faster to compute.

Entropy:

It is also one of the measure used in Decision trees. Entropy comes from Information theory it measures unpredictibility or disorder in a set. The fromula used for calculating Entropy is: [Entropy=-sigma(1 to n)probability of class(Pi)log base2 probability of class(Pi)] if all examples are of the same class then entropy is 0 It has more theoretical and uses information gain.

Coming to the code implementaion the libraries used are: 1.Pandas --> Used for data manipulation

2.Numpy -->Used for Numerical computations

3.Matplotlib.pyplot --> Data Visualization and used for plotting tree

4.Scikit Learn --> It is used for Preprocessing Data, Splitting Data, importing DecisionTreeClassifier(), Evaluating Model accuracy, Visualizing model. scikit learn(sklearn) is a powerful machine learning libray in python.

Methods Used in the implementation is:

Pd.read_csv('file_path.csv') This method is used to load the dataset from comma seperated value into pandas dataframe.

LabelEncoder() This encodes string labels into integers (yes,no to 1 or 0)

train_test_split(x,hy,test_size=0.2,random_state=42) This is used for splitting data into 80% for training and 20% for the testing

model.fit() This trains the decision tree model using the training data

predict() This makes predictions on test set

accuracy_score() This method compares actual test and predicted values and returns a value between 0 and 1.

classification_report() This method returns precision, recall,f1score,support for each class. It helps to understand the model performance in detail.

plot_tree() This method visualizes the decision tree structure.

Dataset: The dataset contains 1000 records and 21 columns. It is related to credit risk assessment the dataset contains data to classify wether a person has a good or bad credit risk.

Based on features such as age,savings,employment,credit_history and more..The target variable is credit_risk which is binary.

**OUTPUT**
Model accuracy (entropy): 0.745000
Classification Report:                precision    recall  f1-score   support

           0       0.68      0.25      0.37        59
           1       0.75      0.95      0.84       141

    accuracy                           0.74       200
   macro avg       0.72      0.60      0.61       200
weighted avg       0.73      0.74      0.70       200

![Image](https://github.com/user-attachments/assets/2a16dc80-fd72-418a-92d2-42d23d51711a)


Accuracy Score(gini) is: 0.745000
Classification report is:                precision    recall  f1-score   support

           0       0.67      0.27      0.39        59
           1       0.76      0.94      0.84       141

    accuracy                           0.74       200
   macro avg       0.71      0.61      0.61       200
weighted avg       0.73      0.74      0.71       200

![Image](https://github.com/user-attachments/assets/2cfd413a-1607-49e2-a9aa-c77a9697e67a)




