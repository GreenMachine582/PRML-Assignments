# PRML-MachineLearning

```python
import machine_learning
raw_dataset = loadDataset(dataset_name)

processed_dataset = processData(raw_dataset)

dataset = extractFeatures(processed_dataset)

saveDataset(dataset)

datasets = splitDataset(dataset)

model = trainModel(model_type, datasets)

resultAnalysis(model, datasets, dataset)
```