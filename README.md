# ML_Avila
Machine learning project to classify the authors based on writing style of the Avila bible


2 Methods
  The following classifiers were implemented and compared,
  ➢ Neural Networks
  ➢ Support Vector Machines
  ➢ Decision Tree Classifier

3 Evaluate
  The labeled data was split into 80% Training data and 20% Testing data.
  The Model was trained on the train set and the accuracy score metric from the sklearn package was used to evaluate the accuracy on the test set.
  This accuracy score was then compared to identify the best classifier amongst the three.
  
 4 Decision Tree Classifier
Decision tree classifier was for its advantages over other classifiers and through comparison of prediction accuracy of the test data.
Gridsearch CV from the scikit package was used to identify the best possible parameters. The following parameters were obtained.
Optimal parameters: {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 32, 'max_features': 'auto', 'min_samples_split': 2, 'splitter': 'best'}
The balanced error rate was calculated to be 0.09 on the test data.
The unlabeled data was imported into the code and missing values were handled using the same method as the labeled data.
This final model was used to predict the unlabeled data and the predictions are saved in Output folder as “Avila_Results.csv”.
