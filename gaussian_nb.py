import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold




df = pd.read_csv('dataset.csv')

# Encode categorical variables
label_encoder = LabelEncoder()


# List of categorical columns to encode
categorical_columns = [
    'Genes in mother\'s side', 'Inherited from father', 'Maternal gene', 'Paternal gene',
    'Parental consent', 'Follow-up', 'Gender', 'Birth asphyxia',
    'Autopsy shows birth defect (if applicable)', 'Place of birth',
    'Folic acid details (peri-conceptional)', 'H/O serious maternal illness',
    'H/O radiation exposure (x-ray)', 'H/O substance abuse', 'Status', 'Respiratory Rate (breaths/min)', 'Heart Rate (rates/min)',
    'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies',
    'Birth defects', 'Blood test result', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'
]

# Model used
nb = GaussianNB()




# Print df stats

print('-DATASET COUNTS BEFORE DROPPING UNKNOWNS-\n')

print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())


# Train models without unknown data

print('\n\n-------------------------REMOVING UNKNOWNS IN ALL COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df.replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

print('-DATASET COUNTS AFTER DROPPING ALL UNKNOWNS-\n')
print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())

# Prepare catgorical data

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column].astype(str))


# Train model for the second-to-last column

# Extract features and target
X = df.iloc[:, :-2]
y = df.iloc[:, -2]

# Training and Testing of the model
scores = cross_val_score(nb, X, y, cv=10, scoring='f1_macro')  # evaluate the model scores

meanF1 = np.mean(scores)

print('\n-MODEL PREFORMANCE-\n')

print("F1 Score Macro-Averaged\n Second-to-Last Column=", meanF1)


# Train model for the last column
y = df.iloc[:, -1]

# Training and Testing of the model
scores = cross_val_score(nb, X, y, cv=10, scoring='f1_macro')  # evaluate the model scores

meanF1 = np.mean(scores)

print("\n Last Column=", meanF1)
