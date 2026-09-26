import pandas as pd
import os

print("Reading Train S1 countries...")
train_s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', usecols=['country'], dtype=str)
print("Train S1 top countries:")
print(train_s1['country'].value_counts(dropna=False).head(15))

print("\nReading Test S1 countries...")
test_s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', usecols=['country'], dtype=str)
print("Test S1 top countries:")
print(test_s1['country'].value_counts(dropna=False).head(15))
