# PRML-MachineLearning

```python
import os
import PRML

ROOT_DIR = os.path.dirname(__file__)
dataset_name = 'Fashion-MNIST'
config = PRML.Config(ROOT_DIR, dataset_name)
ml = PRML.MachineLearning(config)
ml.main()
```
#### Callable Functions
```python
dataset = ml.loadDataset()
ml.saveDataset(dataset)

dataset, X, y = ml.processData(dataset)

dataset, X, y = ml.extractFeatures(dataset, X, y)

ml.exploratoryDataAnalysis(dataset, X, y)

X_train, X_test, y_train, y_test = ml.splitDataset(X, y)

model = ml.trainModel(X_train, y_train)

ml.resultAnalysis(model, X, X_test, y_test)

ml.saveModel(model)
model = ml.loadModel()
```
