# PRML-MachineLearning

```python
import machine_learning as ml

ml.main()
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
