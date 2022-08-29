# PRML-MachineLearning

```python
import os
import PRML

ROOT_DIR = os.path.dirname(__file__)

config = PRML.Config(ROOT_DIR)
ml = PRML.MachineLearning(config, dataset_name)
ml.main()
```
#### Callable Functions
```python
dataset = ml.loadDataset(dataset_name)
ml.saveDataset(dataset_name, dataset)

dataset, X, y = ml.processData(dataset)
dataset, X, y = ml.extractFeatures(dataset)

PRML.saveDataset(dataset)

X_train, X_test, y_train, y_test = ml.splitDataset(X, y)

model = ml.loadModel(model_dir)
# -- or --
model = ml.trainModel(X_train, y_train)

ml.resultAnalysis(model, X, X_test, y_test)

ml.saveModel(model_dir, model)
```
#### Config
```python
# Check and change the default values of config, eg.
config.dataset_dir = f'{ROOT_DIR}\\datasets\\'

# Dataset names without a .csv will be fetched from OpenML - 'https://www.openml.org'
dataset_name = 'Fashion-MNIST' or 'Fashion-MNIST.csv'
```
