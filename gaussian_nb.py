import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder




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

all_x_columns= [
    'Patient Age', "Genes in mother's side", 'Inherited from father', 'Maternal gene', 'Paternal gene',
    'Blood cell count (mcL)', "Mother's age", "Father's age", 'Status', 'Respiratory Rate (breaths/min)',
    'Heart Rate (rates/min)', 'Parental consent', 'Follow-up', 'Gender', 'Birth asphyxia',
    'Autopsy shows birth defect (if applicable)', 'Place of birth', 'Folic acid details (peri-conceptional)',
    'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse',
    'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies', 'No. of previous abortion',
    'Birth defects', 'White Blood cell count (thousand per microliter)', 'Blood test result',
    'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5',
]

# Model used
nb = GaussianNB()




# Print df stats

print('-DATASET COUNTS BEFORE DROPPING UNKNOWNS-\n')

print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())


# Train models without unknown targets

print('\n\n-------------------------REMOVING UNKNOWNS IN TARGET COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df['Genetic Disorder'].replace('Unknown', np.nan, inplace=True)
df['Disorder Subclass'].replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

print('-DATASET COUNTS AFTER DROPPING TARGET UNKNOWNS-\n')
print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())

# Prepare catgorical data

df_encoded=df.copy()
for column in all_x_columns:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))


# Train model for the second-to-last column

# Extract features and target
X = df_encoded.iloc[:, :-2]
y = df_encoded.iloc[:, -2]

# Training and Testing of the model
scores = cross_val_score(nb, X, y, cv=10, scoring='f1_macro')  # evaluate the model scores

meanF1 = np.mean(scores)

print('\n-MODEL PREFORMANCE-\n')

print("F1 Score Macro-Averaged\n Second-to-Last Column=", meanF1)


# Train model for the last column
y = df_encoded.iloc[:, -1]

# Training and Testing of the model
scores = cross_val_score(nb, X, y, cv=10, scoring='f1_macro')  # evaluate the model scores

meanF1 = np.mean(scores)

print("\n Last Column=", meanF1)


# Train models without unknown data

print('\n\n-------------------------REMOVING UNKNOWNS IN ALL COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df.replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

print('-DATASET COUNTS AFTER DROPPING ALL UNKNOWNS-\n')
print(df.iloc[:, -2].value_counts())
print(df.iloc[:, -1].value_counts())

# Prepare catgorical data

df_encoded=df.copy()
for column in categorical_columns:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))


# Train model for the second-to-last column

# Extract features and target
X = df_encoded.iloc[:, :-2]
y = df_encoded.iloc[:, -2]

# Training and Testing of the model
scores = cross_val_score(nb, X, y, cv=10, scoring='f1_macro')  # evaluate the model scores

meanF1 = np.mean(scores)

print('\n-MODEL PREFORMANCE-\n')

print("F1 Score Macro-Averaged\n Second-to-Last Column=", meanF1)


# Train model for the last column
y = df_encoded.iloc[:, -1]

# Training and Testing of the model
scores = cross_val_score(nb, X, y, cv=10, scoring='f1_macro')  # evaluate the model scores

meanF1 = np.mean(scores)

print("\n Last Column=", meanF1)
