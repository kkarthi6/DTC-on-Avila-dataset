# To load pandas
import pandas as pd
# To load numpy
import numpy as np
# To import the classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# To measure accuracy
from sklearn import metrics
from sklearn import model_selection
# from sklearn.metrics import confusion_matrix
# To import the scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
# To display a decision tree
from sklearn.tree import plot_tree
# To support plots
import matplotlib.pyplot as plt
# For data preprocessing
from sklearn.utils import resample

class DummyScaler:
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data
def create_scaler_dummy():
    return DummyScaler()
   
def create_scaler_standard():
    return StandardScaler()
def create_scaler_minmax():
    return MinMaxScaler()
def crete_scaler_binarizer():
    return Binarizer()

# You can choose a scaler (just one should be uncommented):
create_scaler = create_scaler_dummy
# create_scaler = create_scaler_standard
# create_scaler = create_scaler_minmax
# create_scaler = create_scaler_binarizer

# The Decision tree classifier
def create_model_decision_tree():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # model = DecisionTreeClassifier(min_samples_split=5, random_state=seed, presort=True)
    model = DecisionTreeClassifier(min_samples_split=5, random_state=seed)
    return model
create_model = create_model_decision_tree
seed = 520
np.set_printoptions(precision=3)

print('Load the data')

#data = pd.read_csv(r'C:\Users\kkarthi6\Documents\IEE520\Avila_cleaned_data(Labeled).csv')
# Use data from the data folder located here
# https://github.com/kkarthi6/ML_Avila/tree/main/Data
data.head()

#import pandas_profiling as pdp

#Creating the Exploratory Data Analysis of our dataset
#report = pdp.ProfileReport(data, title='Pandas Profiling Report', style={'full_width':True})

# Separate majority and minority classes
data_majority = data[data.y==0]
data_minority = data[data.y==1]
 
# Upsample minority class
data_minority_upsampled = resample(data_minority,replace=True,n_samples=6937,random_state=123)
 
# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority,data_minority_upsampled])
 
# Display new class counts
data_upsampled.y.value_counts()

vals = data_upsampled.values
y = vals[:, -1]
X = vals[:, :-1]

print('Features:')
print(X)
print('Targets:')
print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'None', strategy = 'most_frequent')
imputer = imputer.fit(X[:, 0:12])
X[:, 0:12] = imputer.transform(X[:, 0:12])
X[:, 0:12] = X[:, 0:12].astype('int')  
y = y.astype('int')

print('Train the model and predict')
scaler = create_scaler()
model = create_model()
model.fit(X, y)
y_hat = model.predict(X)

print('Model evaluation (train)')
print('Accuracy:')
print(metrics.accuracy_score(y, y_hat))
print('Classification report:')
print(metrics.classification_report(y, y_hat))

# print('Confusion matrix (train)')
# cm = confusion_matrix(y, y_hat)
# print(cm)

print('Confusion matrix')
df = pd.DataFrame({'y_Actual':y, 'y_Predicted':y_hat})
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

print('Cross-validation')
np.random.seed(seed)
y_prob = np.zeros(y.shape)
y_hat = np.zeros(y.shape)

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)

# Cross-validation
for train, test in kfold.split(X, y):
    # Train classifier on training data, predict test data
    
    # Scaling train and test data
    # Train scaler on training set only
    scaler.fit(X[train])
    X_train = scaler.transform(X[train])
    X_test = scaler.transform(X[test])
    
    model = create_model()
    model.fit(X_train, y[train])
    y_prob[test] = model.predict_proba(X_test)[:, 1]
    y_hat[test] = model.predict(X_test)

print('Model evaluation (CV)')
print('Accuracy:')
print(metrics.accuracy_score(y, y_hat))
print('Classification report:')
print(metrics.classification_report(y, y_hat))

print('Confusion matrix')
df = pd.DataFrame({'y_Actual':y, 'y_Predicted':y_hat})
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

print('ROC curve')

def plot_roc_curve(y_true, y_prob):
    # ROC curve code here is for 2 classes only
    if len(np.unique(y)) == 2: 
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


plot_roc_curve(y, y_prob)

print('Grid Search for Hyperparameters')

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=520)
scaler = create_scaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model.fit(X_train, y_train)

# Here we should use specific classifier, because of the hyperparameters
# model = model_selection.GridSearchCV(DecisionTreeClassifier(random_state=seed, presort=True),
model = model_selection.GridSearchCV(DecisionTreeClassifier(random_state=seed),
                         cv=5,
                         n_jobs=-1,
                         # iid=True,
                         param_grid={
                            'min_samples_split': range(2,3),
                            'max_depth': range(30, 35),
                            'min_samples_leaf' : range(1,5),
                            'criterion': ['gini', 'entropy'],
                            'splitter' : ['best', 'random'],
                            'max_features' : ['auto', 'sqrt', 'log2'],
                            'class_weight' : ['balanced']
                         })


model.fit(X_train, y_train)
print('Optimal parameters:', model.best_params_)
y_test_hat = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

print('Model evaluation (Optimal Hyperparameters)')
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_test_hat))
print('Balanced error rate:')
print(1-metrics.balanced_accuracy_score(y_test, y_test_hat))

# print('Classification report:')
# print(metrics.classification_report(y_test, y_test_hat))

# print('Confusion matrix')
# df = pd.DataFrame({'y_Actual':y, 'y_Predicted':y_hat})
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# print (confusion_matrix)

# print('ROC curve (Optimal Hyperparameters)')
# plot_roc_curve(y_test, y_test_prob)

# tree = model.best_estimator_
# plot_tree(tree, filled=True)
# plt.show()

# Re-training the model on labeled data
# Optimal parameters: {'class_weight': 'balanced', 'criterion': 'gini',
# 'max_depth': 28, 'max_features': 'auto', 'min_samples_split': 2, 'splitter': 'best'}
# def create_model_decision_tree():
#     # You can find the full list of parameters here:
#     # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#     # model = DecisionTreeClassifier(min_samples_split=5, random_state=seed, presort=True)
#     model = DecisionTreeClassifier(min_samples_split=2,
#                                    class_weight= 'balanced',
#                                    max_depth= 28,
#                                    criterion = 'gini',
#                                    splitter = 'best',
#                                    max_features= 'auto',
#                                    random_state=seed)
#     return model
# create_model = create_model_decision_tree
# model = create_model()
# model.fit(X, y)

# Predicting the unlabeled values
print('Load the data')
# Use data from the data folder located here
# https://github.com/kkarthi6/ML_Avila/tree/main/Data
#data = pd.read_csv(r'C:\Users\kkarthi6\Documents\IEE520\Avila_cleaned_data(Unlabeled).csv')

vals = data.values
X_ul = vals[:, 1:-1]
print('Features:')
print(X_ul)

imputer = SimpleImputer(missing_values = 'None', strategy = 'most_frequent')
imputer = imputer.fit(X_ul[:, 0:12])
X_ul[:, 0:12] = imputer.transform(X_ul[:, 0:12])
X_ul[:, 0:12] = X_ul[:, 0:12].astype('int')  

y_pred = model.predict(X_ul)
print(y_pred)

results = pd.DataFrame(y_pred)
print(results)

#results.to_csv (r'C:\Users\kkarthi6\Documents\IEE520\Avila_predictions.csv')

























