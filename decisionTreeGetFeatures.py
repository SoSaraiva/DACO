'''Code to get the features importances with which the model should be retrained with'''

# Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score, ParameterGrid
from sklearn.multioutput import MultiOutputClassifier
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Load dataset
dataset= pd.read_csv('dataset.csv')
#print(dataset.columns)

# Replace 'Unknown' with NaN in Target 
dataset['Genetic Disorder'].replace('Unknown', np.nan, inplace=True)
dataset['Disorder Subclass'].replace('Unknown', np.nan, inplace=True)
# Eliminate samples of NaN targets
dataset.dropna(inplace=True)


# Encode categorical variables
label_encoder = LabelEncoder()


# List of categorical columns to encode
categorical_unordered_columns = [
    'Genes in mother\'s side', 'Inherited from father', 'Maternal gene', 'Paternal gene', 'Status',
    'Respiratory Rate (breaths/min)', 'Heart Rate (rates/min)', 'Follow-up',
    'Gender', 'Birth asphyxia', 'Autopsy shows birth defect (if applicable)', 'Place of birth',
    'Folic acid details (peri-conceptional)',
    'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse', 
    'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies',
    'Birth defects', 'Blood test result',
    'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5', 
    'Genetic Disorder', 'Disorder Subclass'
]


quantitative_with_unknowns_or_ordered_columns = [
    'Patient Age', "Mother's age", "Father's age", 'No. of previous abortion',
    'White Blood cell count (thousand per microliter)']


dataset_encoded=dataset.copy()


for column in quantitative_with_unknowns_or_ordered_columns:
    dataset_encoded[column] = label_encoder.fit_transform(dataset_encoded[column].astype(str))
dataset=pd.get_dummies(dataset_encoded, columns=categorical_unordered_columns, drop_first=False)
print(dataset.columns)
# Alocate features and labels

X = dataset.iloc[:, :-12]  # Features
y = dataset.iloc[:, -12:]  # Labels (last two columns encoded in 3+9 columns)


# Split the data into validation, validation and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  random_state=1) # training as 75% 

dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Plot feature importance
feature_importance = dt.feature_importances_
feature_names = X_train.columns
sorted_idx = feature_importance.argsort()

plt.barh(range(len(feature_names)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_names)), feature_names[sorted_idx])
plt.xlabel('Feature Importance')
plt.show()

# Got 'Patient Age', "Mother's age", "Father's age", 'No. of previous abortion', 'White Blood cell count (thousand per microliter)'