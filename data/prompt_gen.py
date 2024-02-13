from datasets import load_dataset

dataset = load_dataset('Multilingual-Perspectivist-NLU/EPIC', split='train')
dataset = dataset.to_pandas()
print(dataset)
