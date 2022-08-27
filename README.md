# PRML-MachineLearning

```python
import machine_learning as ml

config = PRML.Config(ROOT_DIR)
ml = PRML.MachineLearning(config, 'fashion-mnist_test.csv')
ml.main()
```
-- Callable Functions --
```python
raw_dataset = ml.loadDataset(dataset_name)

dataset, X, y = ml.processData(raw_dataset)

PRML.saveDataset(dataset)

X_train, X_test, y_train, y_test = ml.splitDataset(X, y)

model = ml.trainModel(X_train, y_train)

ml.resultAnalysis(X, X_test, y_test)
```
