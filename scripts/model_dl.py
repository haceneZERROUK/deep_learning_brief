import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder,OrdinalEncoder, LabelEncoder, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from utils import build_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


# Bonne mise en forme des noms de colonnes
df = pd.read_csv('dataset.csv')
df.columns = [c.strip().lower() for c in df.columns]

# 2. Strip whitespace from column names
df.columns = df.columns.str.strip()

# 3. Trim whitespace in all string cells
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 4. Convert empty strings to NaN (optional)
df.replace('', np.nan, inplace=True)

# 5. Drop duplicate rows
df.drop_duplicates(inplace=True)
df.totalcharges = pd.to_numeric(df.totalcharges)

# 6. Drop rows with missing values
df.dropna(inplace=True)



numerical_column = [

'tenure', 
'monthlycharges',
'totalcharges'
]


ordinal_column = [
'seniorcitizen',
'gender',
'partner',
'dependents',
'phoneservice',
'paperlessbilling',


]

categorical_column = [

'multiplelines', 
'onlinesecurity',
'onlinebackup', 
'deviceprotection', 
'techsupport',
'streamingtv', 
'streamingmovies', 
'contract', 
'paymentmethod', 


]

target_name = "churn"

# Séparation train/val/test (80/20 puis 20% de train pour val)
X = df.drop(columns=target_name)
y=df[target_name]
binirazer = LabelBinarizer()
y = binirazer.fit_transform(y)
X.shape, y.shape


#train

X_train_0, X_test, y_train_0, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# On prend 20% de X_train pour validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_0, y_train_0, test_size=0.2, random_state=42, stratify=y_train_0
)
X_train.shape, X_test.shape, X_val.shape, X_test.shape, y_val.shape, y_test.shape

# Construction du transformateur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_column),
        ('ord', OrdinalEncoder(), ordinal_column),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_column)
    ]
)

# Fit sur le train uniquement
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test= preprocessor.transform(X_test)


# Vérification des classes
num_classes = len(np.unique(y))


# Encodage des labels en one-hot
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)


model = build_model(X_train,num_classes)


y_train_labels = np.argmax(y_train_cat, axis=1)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weight_dict = dict(enumerate(class_weights))


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_recall',    # surveille la perte de validation
    patience=3,            # tolère 3 époques sans amélioration
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=20,
    batch_size=8,
    class_weight=class_weight_dict,
    verbose=1,
    callbacks = [early_stop]
)

model.save("prev_churn.keras")

test_loss, test_acc,test_auc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nAccuracy sur le test set : {test_acc:.4f}")
print(f"\nauc sur le test set : {test_auc:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=df.churn.unique()))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df.churn.unique(), yticklabels=df.churn.unique())
plt.title("Matrice de Confusion")
plt.xlabel("Classe Prédite")
plt.ylabel("Classe Réelle")
plt.show()