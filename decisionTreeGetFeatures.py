'''Code to get the features importances with which the model should be retrained with'''

# Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, hamming_loss, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

provided_weights_genetic_disorder = {
    'Mitochondrial genetic inheritance disorders': 0.48794813542417026,
    'Single-gene inheritance diseases': 0.6160580705934504,
    'Multifactorial genetic inheritance disorders': 0.8959937939823793
}

provided_weights_subclass_disorder = {
    'Leigh syndrome': 0.740510888236272,
    'Mitochondrial myopathy': 0.7799634288247355,
    'Cystic fibrosis': 0.8257328087770821,
    'Tay-Sachs': 0.8583698121571453,
    'Diabetes': 0.9084058292236937,
    'Hemochromatosis': 0.9319554496592232,
    "Leber's hereditary optic neuropathy": 0.9674738183631628,
    "Alzheimer's": 0.9926303540754696,
    'Cancer': 0.9949576106832161
}



# Load dataset
dataset= pd.read_csv('dataset.csv')

# Replace 'Unknown' with NaN
dataset.replace('Unknown', np.nan, inplace=True)

# Eliminate samples with
dataset.dropna(inplace=True)



# Encode categorical variables
label_encoder = LabelEncoder()

categorical_columns = [
    'Genes in mother\'s side', 'Inherited from father', 'Maternal gene', 'Paternal gene', 'Status',
    'Respiratory Rate (breaths/min)', 'Heart Rate (rates/min)', 'Follow-up',
    'Gender', 'Birth asphyxia', 'Autopsy shows birth defect (if applicable)', 'Place of birth',
    'Folic acid details (peri-conceptional)',
    'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse', 
    'Assisted conception IVF/ART', 'History of anomalies in previous pregnancies',
    'Birth defects', 'Blood test result',
    'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'
]

quantitative_with_unknowns_or_ordered_columns = [
    'Patient Age', "Mother's age", "Father's age", 'No. of previous abortion',
    'White Blood cell count (thousand per microliter)']


for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column].astype(str))

# Create a copy for encoding
dataset_encoded = dataset.copy()


# Encode target variables 'Genetic Disorder' and 'Disorder Subclass'
# Create mapping
genetic_disorder_mapping = {label: i for i, label in enumerate(dataset_encoded['Genetic Disorder'].unique()) if pd.notna(label)}
disorder_subclass_mapping = {label: i for i, label in enumerate(dataset_encoded['Disorder Subclass'].unique())}

# Replace each original value with its corresponding encoded value as per the mapping
dataset_encoded['Genetic Disorder'] = dataset_encoded['Genetic Disorder'].map(genetic_disorder_mapping)
dataset_encoded['Disorder Subclass'] = dataset_encoded['Disorder Subclass'].map(disorder_subclass_mapping)


# Correspond weights to classes
# Get values mapped
encoded_values_genetic_disorder = dataset_encoded['Genetic Disorder'].unique()
encoded_values_disorder_subclass = dataset_encoded['Disorder Subclass'].unique()

# Inverse mappings to get back the original names
inverse_genetic_disorder_mapping = {i: label for label, i in genetic_disorder_mapping.items()}
inverse_disorder_subclass_mapping = {i: label for label, i in disorder_subclass_mapping.items()}

# Map encoded values back to original names
names_genetic_disorder = [inverse_genetic_disorder_mapping[i] for i in encoded_values_genetic_disorder]
names_disorder_subclass = [inverse_disorder_subclass_mapping[i] for i in encoded_values_disorder_subclass]

# Associate weights with encoded values
weights_genetic_disorder = {encoded_value: provided_weights_genetic_disorder[name] for encoded_value, name in zip(encoded_values_genetic_disorder, names_genetic_disorder)}
weights_disorder_subclass = {encoded_value: provided_weights_subclass_disorder[name] for encoded_value, name in zip(encoded_values_disorder_subclass, names_disorder_subclass)}


# Combine both class weights into a dictionary
class_weights = {'Genetic Disorder': weights_genetic_disorder, 'Disorder Subclass': weights_disorder_subclass}


# Allocate features and labels
X = dataset_encoded.drop(columns=['Genetic Disorder', 'Disorder Subclass'])
y = dataset_encoded[['Genetic Disorder', 'Disorder Subclass']]

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

# Got 'Patient Age', "Mother's age", "Father's age", 'No. of previous abortion', 'White Blood cell count (thousand per microliter), No. of previous abortion'