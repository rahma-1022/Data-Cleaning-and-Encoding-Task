import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

df = sns.load_dataset('tips')

df.iloc[0, 0] = np.nan 
df.iloc[5, 2] = np.nan  

print("----- Original data -----")
print(df.head())

target_col = 'day'
col_position = df.columns.get_loc(target_col)

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[[target_col]])

encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([target_col]))

df_cleaned = pd.concat([df.iloc[:, :col_position], encoded_df, df.iloc[:, col_position+1:]], axis=1)

print(f"\n----- After modifying '{target_col}' -----")
print(df_cleaned.head())

missing_info = df_cleaned.isnull().sum()
missing_pct = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100

print("\n----- Analysis of missing values-----")
analysis_df = pd.DataFrame({'Missing Values': missing_info, 'Percentage (%)': missing_pct})
print(analysis_df)

num_imputer = SimpleImputer(strategy='mean')
df_cleaned['total_bill'] = num_imputer.fit_transform(df_cleaned[['total_bill']]).ravel()

cat_imputer = SimpleImputer(strategy='most_frequent')
df_cleaned['sex'] = cat_imputer.fit_transform(df_cleaned[['sex']]).ravel()

print("\n----- Final data -----")
print(df_cleaned.head())
print("\n missing values:")
print(df_cleaned.isnull().sum().sum())