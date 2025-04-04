import pandas as pd

print(f"Importing datasets...")
from datasets import load_dataset

print(f"Importing evaluator...")
from evaluator import Evaluator

print(f"Loading dataset...")
splits = load_dataset("ecaplan/splits", "splits")
splits = splits['train']

sample_size = 100

# splits = splits[:sample_size]
print(f"Dataset loaded. Processing...")

# convert the dataset to a pandas dataframe
df = pd.DataFrame(splits)
small_df = df.sample(sample_size, random_state=42)

# make some mock theories
theories = ['Catholics use more commas, Black people use more quotes, Teachers use more dashes, Construction workers use fewer words, Hindus use more foreign words, and Jewish people use more symbols.'] * sample_size
small_df['theory'] = theories

print(f"Dataframe processed. Creating evaluator...")
print(f"Dataframe: {small_df}")

e = Evaluator(small_df, test=False, batch_size=1)

print(f"Evaluator created. Evaluating...")

e.evaluate()

print(f"Evaluation complete. Predictions: {list(e.matching_df['prediction'])}")
print(f"Generating reports...")

split_reports = e.get_all_split_reports()

print(f"Reports generated: {split_reports}")

accuracy = e.get_accuracy()

print(f"Accuracy: {accuracy}")