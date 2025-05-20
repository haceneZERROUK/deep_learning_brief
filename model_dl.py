import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder,OrdinalEncoder, LabelEncoder


# Bonne mise en forme des noms de colonnes
df = pd.read_csv('dataset.csv')
df.columns = [c.strip().lower() for c in df.columns]