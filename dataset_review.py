import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')

df.info()

print('\n-DATASET TARGET COLUMN COUNTS BEFORE DROPPING UNKNOWNS-\n')

print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())

print('\n\n-------------------------REMOVING UNKNOWNS IN TARGET COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df['Genetic Disorder'].replace('Unknown', np.nan, inplace=True)
df['Disorder Subclass'].replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

print('-DATASET COUNTS AFTER DROPPING TARGET UNKNOWNS-\n')
print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())


print('\n\n-------------------------REMOVING UNKNOWNS IN ALL COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df.replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

print('-DATASET COUNTS AFTER DROPPING ALL UNKNOWNS-\n')
print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())
