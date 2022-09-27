# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 18:08:38 2022

@author: Leigh Hill u3215513
"""

from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

breast_cancer_dataset = load_breast_cancer()
cancer = breast_cancer_dataset

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_dataset.data, breast_cancer_dataset.target,
                                                    test_size=0.2, random_state=201)

pipeline = Pipeline([('scaler', StandardScaler()),
                     ('svc', SVC(random_state=0))])

  
# defining parameter range # SUPER IMPORTANT TO NOTE: in estimators we call SVC() svc,
# each param needs svc__ added to it for the grid_pipeline fit to work
param_grid = {'svc__C': [0.1, 1, 10, 100, 1000],
              'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'svc__kernel': ['rbf']}

cv_results = GridSearchCV(pipeline, param_grid)
cv_results.fit(breast_cancer_dataset.data, breast_cancer_dataset.target)

print(cv_results.best_params_)

# fit
estimator = cv_results.best_estimator_
estimator.fit(X_train, y_train)
print("Score - %.4f%s" % (estimator.score(X_train, y_train) * 100, "%"))
y_pred = estimator.predict(X_test)

print(breast_cancer_dataset.target_names)
print(metrics.classification_report(y_test, y_pred, target_names=breast_cancer_dataset.target_names))

print("Accuracy - %.4f%s" % (metrics.accuracy_score(y_test, y_pred) * 100, "%"))
print("Precision - %.4f%s" % (metrics.precision_score(y_test, y_pred) * 100, "%"))
print("Recall - %.4f%s" % (metrics.recall_score(y_test, y_pred) * 100, "%"))
print("F1 - %.4f%s" % (metrics.f1_score(y_test, y_pred) * 100, "%"))
