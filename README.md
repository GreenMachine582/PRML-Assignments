# PRML-MachineLearning

```python
import os
import machine_learning

local_dir = os.path.dirname(__file__)
dataset_name = 'Fashion-MNIST'

config = machine_learning.Config(local_dir, dataset_name)
ml = machine_learning.MachineLearning(config)

dataset = ml.loadDataset()
ml.saveDataset(dataset)

dataset, X, y = ml.processData(dataset)

dataset, X, y = ml.extractFeatures(dataset, X, y)

ml.exploratoryDataAnalysis(dataset, X, y)

X_train, X_test, y_train, y_test = ml.splitDataset(X, y)

model = ml.trainModel(X_train, y_train)

ml.resultAnalysis(model, X, X_test, y_test)

model = ml.loadModel()
ml.saveModel(model)
```
