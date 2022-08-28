# PRML-MachineLearning

```python
import PRML

config = PRML.Config(ROOT_DIR)
ml = PRML.MachineLearning(config, 'fashion-mnist_test.csv')
ml.main()
```
-- Callable Functions --
```python
dataset = ml.loadDataset(dataset_name)
ml.saveDataset(dataset_name, dataset)

dataset, X, y = ml.processData(dataset)
dataset, X, y = ml.extractFeatures(dataset)

PRML.saveDataset(dataset)

X_train, X_test, y_train, y_test = ml.splitDataset(X, y)

models = ml.trainModel(X_train, y_train)

ml.resultAnalysis(models, X, X_test, y_test)

ml.saveModel(model_dir, model)
ml.loadModel(model_dir)
```
