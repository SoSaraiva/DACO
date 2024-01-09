import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder




df = pd.read_csv('dataset.csv')

# Encode categorical variables
label_encoder = LabelEncoder()

all_x_columns= [
    'Patient Age', "Genes in mother's side", 'Inherited from father', 'Maternal gene', 'Paternal gene',
    'Blood cell count (mcL)', "Mother's age", "Father's age", 'Status', 'Respiratory Rate (breaths/min)',
    'Heart Rate (rates/min)', 'Follow-up', 'Gender', 'Birth asphyxia',
    'Autopsy shows birth defect (if applicable)', 'Place of birth', 'Folic acid details (peri-conceptional)',
    'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse',
    'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies', 'No. of previous abortion',
    'Birth defects', 'White Blood cell count (thousand per microliter)', 'Blood test result',
    'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'
]

# Train models without unknown targets

print('\n\n-------------------------REMOVING UNKNOWNS IN TARGET COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df['Genetic Disorder'].replace('Unknown', np.nan, inplace=True)
df['Disorder Subclass'].replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

# Prepare catgorical data

df_encoded=df.copy()
for column in all_x_columns:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))


# Train model for the second-to-last column

# Extract features and target
X = df_encoded.iloc[:, :-2]
y = df_encoded.iloc[:, -2]

# Split the data into validation, validation and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) 

print('\n-MODEL PREFORMANCE-\n')

print("Second-to-Last Column:\n")

print(classification_report(y_test, y_pred))


# Train model for the last column
y = df_encoded.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) 

print("\nLast Column:\n")

print(classification_report(y_test, y_pred))


# Multi-label classification
y = df_encoded.iloc[:, -2:]

if y.nunique().any() > 2:
    y = pd.get_dummies(y, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) 


# First try on feature Importances
feature_importances_cat = rf.feature_importances_
print (feature_importances_cat)
import matplotlib.pyplot as plt

feature_names = X.columns 
plt.barh(feature_names, feature_importances_cat)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Random Forest Feature Importances')
plt.show()



print("\nMulti-layer classifier:\n")

print(classification_report(y_test, y_pred, output_dict=True))


# Train models without unknown data

print('\n\n-------------------------REMOVING UNKNOWNS IN ALL COLUMNS---------------------------\n')

# Replace 'Unknown' with NaN
df.replace('Unknown', np.nan, inplace=True)

df.dropna(inplace=True)

# Prepare catgorical data

df_encoded=df.copy()
for column in all_x_columns:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column].astype(str))


# Train model for the second-to-last column

# Extract features and target
X = df_encoded.iloc[:, :-2]
y = df_encoded.iloc[:, -2]

# Split the data into validation, validation and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) 

print('\n-MODEL PREFORMANCE-\n')

print("Second-to-Last Column:\n")

print(classification_report(y_test, y_pred))


# Train model for the last column
y = df_encoded.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

# Model used
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) 

print("\nLast Column:\n")

print(classification_report(y_test, y_pred))



