#!/usr/bin/env python
# coding: utf-8

# <div style="background: linear-gradient(to right, #4F46E5, #7C3AED); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
#     <h1 style="color: white; font-size: 2.5em; margin-bottom: 15px;">Customer Churn Prediction with Deep Learning</h1>
#     <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1em; line-height: 1.6;">
#         Using neural networks to predict customer behavior in the telecom industry
#     </p>
#     <div style="margin-top: 20px;">
#         <span style="background: rgba(255, 255, 255, 0.2); color: white; padding: 8px 16px; border-radius: 20px; margin-right: 10px;">Telecom</span>
#         <span style="background: rgba(255, 255, 255, 0.2); color: white; padding: 8px 16px; border-radius: 20px; margin-right: 10px;">Data Science</span>
#         <span style="background: rgba(255, 255, 255, 0.2); color: white; padding: 8px 16px; border-radius: 20px;">ML</span>
#     </div>
# </div>
# 
# <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 30px;">
#     <h2 style="color: #4F46E5; margin-bottom: 20px;">Project Overview</h2>
#     <p style="color: #374151; font-size: 1.1em; line-height: 1.6;">
#         As Data Scientists at a Digital Services Company specializing in helping telecom operators reduce subscriber loss, 
#         you've been assigned to a new client. <span style="color: #4F46E5; font-weight: 500;">TelcoNova</span> wants to anticipate 
#         customer departures (<span style="color: #4F46E5; font-weight: 500;">churn</span>) to optimize their retention campaigns.
#     </p>
# </div>
# 
# <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px;">
#     <div style="background: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 4px solid #4F46E5;">
#         <h3 style="color: #1F2937; margin-bottom: 10px;">📅 Timeline</h3>
#         <p style="color: #4B5563;">3 days to deliver a working prototype</p>
#     </div>
#     <div style="background: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 4px solid #4F46E5;">
#         <h3 style="color: #1F2937; margin-bottom: 10px;">🔄 Dataset</h3>
#         <p style="color: #4B5563;">Pre-cleaned Telco Customer Churn data</p>
#     </div>
#     <div style="background: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 4px solid #4F46E5;">
#         <h3 style="color: #1F2937; margin-bottom: 10px;">🎯 Main Goal</h3>
#         <p style="color: #4B5563;">Predict customers likely to churn</p>
#     </div>
# </div>
# 
# <div style="position: relative; margin-bottom: 30px;">
#     <img src="https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" 
#          style="width: 100%; height: 300px; object-fit: cover; border-radius: 10px;">
#     <div style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); padding: 15px; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;">
#         <p style="color: white; text-align: center; margin: 0;">
#             Using neural networks to predict customer behavior in the telecom industry
#         </p>
#     </div>
# </div>
# 
# <div style="background: #F8FAFC; padding: 25px; border-radius: 10px; border: 1px solid #E2E8F0;">
#     <h2 style="color: #1F2937; margin-bottom: 20px;">Project Goals</h2>
#     <ul style="list-style-type: none; padding: 0;">
#         <li style="display: flex; align-items: center; margin-bottom: 15px;">
#             <span style="background: #4F46E5; color: white; width: 24px; height: 24px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">1</span>
#             <span style="color: #4B5563;">Early identification of customers likely to churn</span>
#         </li>
#         <li style="display: flex; align-items: center; margin-bottom: 15px;">
#             <span style="background: #4F46E5; color: white; width: 24px; height: 24px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">2</span>
#             <span style="color: #4B5563;">Optimization of retention campaign targeting</span>
#         </li>
#         <li style="display: flex; align-items: center;">
#             <span style="background: #4F46E5; color: white; width: 24px; height: 24px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">3</span>
#             <span style="color: #4B5563;">Analysis of key factors influencing customer departures</span>
#         </li>
#     </ul>
# </div>

# <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 30px 0;">
#     <h2 style="color: #4F46E5; margin-bottom: 20px;">1. Data Loading and Initial Exploration</h2>
#     <p style="color: #374151; font-size: 1.1em; line-height: 1.6;">
#         In this section, we'll load the telecom customer dataset and perform initial exploration to understand its structure and contents.
#         This helps us get familiar with the data before preprocessing.
#     </p>
# </div>

# In[1]:


# Import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras import layers, models

from scikeras.wrappers import KerasClassifier

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Reducing tensorFlow log verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[2]:


# Load the dataset
df = pd.read_csv('Dataset.csv')

# Display basic information about the dataset
print(f"The data set has : {df.shape[0]} number of rows and {df.shape[1]} number of columns ")
print("\nDataset Info:")
df.info()

# Display first few rows
print("\nFirst few rows of the dataset:")
df.head(2)


# <div style="background: #EEF2FF; padding: 20px; border-radius: 10px; border-left: 4px solid #4F46E5; margin: 20px 0;">
#     <h3 style="color: #1F2937; margin-bottom: 10px;">📊 Initial Data Insights</h3>
#     <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
#         <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
#             <p style="color: #6B7280; margin-bottom: 5px;">Total Customers</p>
#             <p style="color: #4F46E5; font-size: 1.5em; font-weight: bold;">7,043</p>
#         </div>
#         <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
#             <p style="color: #6B7280; margin-bottom: 5px;">Features</p>
#             <p style="color: #4F46E5; font-size: 1.5em; font-weight: bold;">20</p>
#         </div>
#         <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
#             <p style="color: #6B7280; margin-bottom: 5px;">Data Types</p>
#             <p style="color: #4F46E5; font-size: 1.5em; font-weight: bold;">18 cat, 3 num</p>
#         </div>
#     </div>
# </div>

# <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 30px 0;">
#     <h2 style="color: #4F46E5; margin-bottom: 20px;">2. Data Preprocessing</h2>
#     <p style="color: #374151; font-size: 1.1em; line-height: 1.6;">
#         Before building our model, we need to clean and prepare the data. This includes handling missing values,
#         encoding categorical features, and scaling numerical features to ensure optimal model performance.
#     </p>
# </div>

# In[3]:


# Data Cleaning Steps
# 1. Strip whitespace from column names
df.columns = df.columns.str.strip()

# 2. Trim whitespace in all string cells
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 3. Convert empty strings to NaN (optional)
df.replace('', np.nan, inplace=True)

# 4. Show the number of duplicates and number of rows before dropping
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
print(f"Number of rows before dropping duplicates: {df.shape[0]}")

# 5. Drop duplicate rows
df.drop_duplicates(inplace=True)

# 6. Show the number of rows after dropping duplicates
print(f"Number of rows after dropping duplicates: {df.shape[0]}")

# 7. Show the number of missing values for the columns which have them
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(f"\nNumber of missing values per column if any:\n{missing_values}")

# 8. Drop rows with missing values
df.dropna(inplace=True)

# 9. (Optional) See the result
print("\nAfter cleaning, the dataset has:")
print(f"Number of rows: {df.shape[0]}")

# 10. Display the first few rows of the cleaned dataset
print("\nFirst few rows of the cleaned dataset:")
print(df.head(2))
# 11. Display the data types of the columns
print("\nData types of the columns:")
print(df.dtypes)
# 12. Display the number of unique values in each column
print("\nNumber of unique values in each column:")
print(df.nunique())


# <div style="background: #FEF3C7; padding: 20px; border-radius: 10px; border-left: 4px solid #D97706; margin: 20px 0;">
#     <h3 style="color: #92400E; margin-bottom: 10px;">⚠️ Data Cleaning Insights</h3>
#     <ul style="color: #92400E; margin: 0; padding-left: 20px;">
#         <li>No duplicate rows were found in the dataset</li>
#         <li>11 missing values were detected in the TotalCharges column and were removed</li>
#         <li>After cleaning, we have 7,032 rows of data (reduced from 7,043)</li>
#         <li>The dataset contains a mix of categorical and numerical features that will need different preprocessing approaches</li>
#     </ul>
# </div>

# In[4]:


# Feature Engineering and Encoding
# Features / target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 1) Map Churn to 0 and 1
y = y.map({"No": 0, "Yes": 1})
# 2) Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# 3) Drop rows with NaN in TotalCharges
df.dropna(subset=["TotalCharges"], inplace=True)    
# 4) Drop customerID
X.drop(columns=["customerID"], inplace=True)
# 5) Convert categorical columns to category dtype
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].astype("category")

# 2) Split into train + held-out test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3) Define your preprocessor
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])


# <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 30px 0;">
#     <h2 style="color: #4F46E5; margin-bottom: 20px;">3. Model Development</h2>
#     <p style="color: #374151; font-size: 1.1em; line-height: 1.6;">
#         In this section, we build and train a neural network model for predicting customer churn.
#         We'll implement K-fold cross-validation to ensure our model generalizes well to unseen data.
#     </p>
# </div>

# In[5]:


# Manual 5‐fold CV on the TRAINING set
from tensorflow.keras import Model, Input, layers
from sklearn.metrics import roc_curve, auc

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_histories = []
all_models    = []
fold_aucs     = []

def create_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    # Split
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Preprocess
    X_tr_p  = preprocessor.fit_transform(X_tr)    # must produce a dense array
    X_val_p = preprocessor.transform(X_val)

    # Build & train
    model = create_model(X_tr_p.shape[1])
    history = model.fit(
        X_tr_p, y_tr,
        validation_data=(X_val_p, y_val),
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )

    # Store
    all_histories.append(history)
    all_models.append(model)

    # Evaluate fold AUC
    y_val_pred = model.predict(X_val_p).ravel()
    fpr, tpr, _ = roc_curve(y_val, y_val_pred)
    fold_auc = auc(fpr, tpr)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold} ROC-AUC: {fold_auc:.3f}")

print(f"\nCV Mean ROC-AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")


# <div style="background: #EEF2FF; padding: 20px; border-radius: 10px; border-left: 4px solid #4F46E5; margin: 20px 0;">
#     <h3 style="color: #1F2937; margin-bottom: 10px;">🧠 Model Architecture Insights</h3>
#     <p style="color: #374151; margin-bottom: 15px;">The neural network architecture is designed for binary classification with multiple hidden layers:</p>
#     <ul style="color: #374151; margin: 0; padding-left: 20px;">
#         <li>3 hidden layers with decreasing number of neurons (64 → 32 → 16) to gradually extract features</li>
#         <li>ReLU activation functions for non-linearity in hidden layers</li>
#         <li>Dropout regularization (0.3 and 0.2) to prevent overfitting</li>
#         <li>Sigmoid activation in the output layer for binary classification probability</li>
#         <li>Adam optimizer with binary cross-entropy loss function</li>
#         <li>Early stopping with patience=3 to prevent overfitting</li>
#     </ul>
# </div>

# In[6]:


# Now fit preprocessor on full TRAINING set & transform TEST set
preprocessor.fit(X_train)
X_test_proc = preprocessor.transform(X_test)

# pick the last fold's model (or average predictions from all_models)
final_model = all_models[-1]

# Plot each fold's training history
for i, history in enumerate(all_histories, start=1):
    h = history.history
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(h['loss'], label='Train Loss')
    plt.plot(h['val_loss'], label='Val Loss')
    plt.title(f'Fold {i} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(h['auc'], label='Train AUC')
    plt.plot(h['val_auc'], label='Val AUC')
    plt.title(f'Fold {i} AUC')
    plt.xlabel('Epoch'); plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Final evaluation on the held-out test set
y_pred_prob    = final_model.predict(X_test_proc).ravel()
y_pred_classes = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred_classes))

# ROC curve on TEST
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1], linestyle='--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Held-Out Test Set')
plt.legend(loc='lower right')
plt.show()


# <div style="background: #FEF3C7; padding: 20px; border-radius: 10px; border-left: 4px solid #D97706; margin: 20px 0;">
#     <h3 style="color: #92400E; margin-bottom: 10px;">📊 Model Performance Analysis</h3>
#     <p style="color: #92400E; margin-bottom: 15px;">Our model demonstrates solid overall performance, but there's room for improvement:</p>
#     <ul style="color: #92400E; margin: 0; padding-left: 20px;">
#         <li><strong>Strengths</strong>: Good overall accuracy (83%) and AUC (0.84)</li>
#         <li><strong>Weaknesses</strong>: Lower recall for the churn class (61%)</li>
#         <li><strong>Concern</strong>: The model is better at identifying non-churning customers (91% recall) than those who will churn (61% recall)</li>
#         <li><strong>Business Impact</strong>: We may miss potential churners, which could be costly in a retention campaign</li>
#     </ul>
# </div>

# <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 30px 0;">
#     <h2 style="color: #4F46E5; margin-bottom: 20px;">5. Model Export and Deployment</h2>
#     <p style="color: #374151; font-size: 1.1em; line-height: 1.6;">
#         After developing and evaluating our model, we need to export it for deployment in a production environment.
#         This involves saving both the model and the preprocessing pipeline to ensure consistent transformation of new data.
#     </p>
# </div>

# In[7]:


import joblib

# 1) Save the final Keras model in the native format
final_model.save('telecom_churn_model.keras')

# 2) Save the preprocessing pipeline
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Model and preprocessing artifacts have been saved successfully!")


# <div style="background: #ECFDF5; padding: 20px; border-radius: 10px; border-left: 4px solid #059669; margin: 20px 0;">
#     <h3 style="color: #065F46; margin-bottom: 10px;">✅ Deployment Success</h3>
#     <p style="color: #065F46; margin-bottom: 15px;">The model has been successfully exported and is ready for deployment. The following artifacts have been saved:</p>
#     <ul style="color: #065F46; margin: 0; padding-left: 20px;">
#         <li><code>telecom_churn_model.keras</code>: The trained neural network model</li>
#         <li><code>preprocessor.pkl</code>: The preprocessing pipeline for feature transformation</li>
#     </ul>
# </div>
