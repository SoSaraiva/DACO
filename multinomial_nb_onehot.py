import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('dataset.csv')

# Encode categorical variables
label_encoder = LabelEncoder()


# List of categorical columns to encode
categorical_unordered_columns = [
    'Genes in mother\'s side', 'Inherited from father', 'Maternal gene', 'Paternal gene',
    'Gender', 'Birth asphyxia',
    'Autopsy shows birth defect (if applicable)', 'Place of birth',
    'Folic acid details (peri-conceptional)', 'H/O serious maternal illness',
    'H/O radiation exposure (x-ray)', 'H/O substance abuse', 'Status',
    'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies',
    'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'
]

quantitative_with_unknowns_or_ordered_columns = ['Patient Age', "Mother's age", "Father's age", 'Respiratory Rate (breaths/min)',
    'Heart Rate (rates/min)', 'Follow-up', 'No. of previous abortion',
    'Birth defects', 'White Blood cell count (thousand per microliter)', 'Blood test result']

# Train models without unknown targets

print('\n\n-------------------------REMOVING UNKNOWNS IN TARGET COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df['Genetic Disorder'].replace('Unknown', np.nan, inplace=True)
df['Disorder Subclass'].replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

# Prepare catgorical data

df_encoded=df.copy()
for column in quantitative_with_unknowns_or_ordered_columns:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))
df_encoded=pd.get_dummies(df_encoded, columns=categorical_unordered_columns, drop_first=False)

# Train model for the second-to-last column

# Extract features and target
X = df_encoded.drop(columns=['Genetic Disorder', 'Disorder Subclass'], axis=1)
y = df_encoded['Genetic Disorder']

# Split the data into validation, validation and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test) 

print('\n-MODEL PREFORMANCE-\n')

print("Second-to-Last Column:\n")

print(classification_report(y_test, y_pred))


# Train model for the last column
y = df_encoded['Disorder Subclass']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test) 

print("\nLast Column:\n")

print(classification_report(y_test, y_pred))


# Train models without unknown data

print('\n\n-------------------------REMOVING UNKNOWNS IN ALL COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df.replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

# Prepare catgorical data

df_encoded=df.copy()
for column in quantitative_with_unknowns_or_ordered_columns:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))
df_encoded=pd.get_dummies(df_encoded, columns=categorical_unordered_columns, drop_first=False)


# Train model for the second-to-last column

# Extract features and target
X = df_encoded.drop(columns=['Genetic Disorder', 'Disorder Subclass'], axis=1)
y = df_encoded['Genetic Disorder']

# Split the data into validation, validation and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test) 

print('\n-MODEL PREFORMANCE-\n')

print("Second-to-Last Column:\n")

print(classification_report(y_test, y_pred))


# Train model for the last column
y = df_encoded['Disorder Subclass']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test) 

print("\nLast Column:\n")

print(classification_report(y_test, y_pred))