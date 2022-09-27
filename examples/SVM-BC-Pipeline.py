from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

breast_cancer_dataset = load_breast_cancer()


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_dataset.data, breast_cancer_dataset.target,
                                                    test_size=0.2, random_state=201)

# Pipline to create Scale and SVC
estimators = [('scaler', StandardScaler()), ('svc', SVC())]
pipe = Pipeline(estimators)
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print("Pipeline SCALER AND SVC ")
print("Score - %.4f%s\n" % (score * 100, "%"))

# principal component analysis, scale and svc
estimators = [('reduce_dim', PCA()), ('scaler', StandardScaler()), ('svm', SVC())]
pipe_2 = Pipeline(estimators)
pipe_2.fit(X_train, y_train)
score = pipe_2.score(X_test, y_test)
print("Pipeline: PCA, SCALER AND SVC ")
print("Score - %.4f%s\n" % (score * 100, "%"))

# Create a svm Classifier
classifiers = [('Linear SVM', SVC(kernel='linear')),
               ('RBF SVM', SVC(kernel='rbf')),
               ('Sigmoid SVM', SVC(kernel='sigmoid')),
               ('Polynomial SVM', SVC(kernel='poly'))]

for name, classifier in classifiers:
    print('\n\t' + name)
    # Train the model using the training sets
    classifier.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = classifier.predict(X_test)

    # Model Accuracy: how often is the classifier correct?

    print("Accuracy - %.4f%s" % (accuracy_score(y_test, y_pred) * 100, "%"))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision - %.4f%s" % (precision_score(y_test, y_pred) * 100, "%"))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall - %.4f%s" % (recall_score(y_test, y_pred) * 100, "%"))
