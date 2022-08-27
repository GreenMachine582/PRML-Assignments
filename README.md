# PRML-MachineLearning

```python
import machine_learning as ml

dataset = 'dataset'  # SKLearn OpenML dataset
dataset = 'datasets\\dataset.csv'  # Dataset directory
ml.main('dataset', model_type='LogisticRegression')
```
--Callable Functions--
```python
raw_dataset = ml.loadDataset(dataset_name)

processed_dataset = ml.processData(raw_dataset)

dataset = ml.extractFeatures(processed_dataset)

ml.saveDataset(dataset)

datasets = ml.splitDataset(dataset)

model = ml.trainModel(model_type, datasets)

ml.resultAnalysis(model, datasets, dataset)
```
