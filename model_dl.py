import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder,OrdinalEncoder, LabelEncoder


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

# 6. Drop rows with missing values
df.dropna(inplace=True)