import pandas as pd

file_path = r"C:\Users\sroes\OneDrive - ilionx Group BV\Bureaublad\Clementine\thesis.csv"
df = pd.read_csv(file_path)

# Initial count
initial_count = len(df)
print(f"Initial number of rows: {initial_count}")
df.info()

# Drop specified columns
columns_to_drop = [
    "personId", "audiogramResultId", "AvlResultId", "DeviceId", "creationDate", "returned"
]
df = df.drop(columns=columns_to_drop)

# Drop missing values
before_dropna = len(df)
df = df.dropna()
print(f"Dropped {before_dropna - len(df)} rows with missing values ({before_dropna} → {len(df)})")

# computed features
df['pta_left_avg'] = (df['left1000'] + df['left2000'] + df['left4000']) / 3
df['pta_right_avg'] = (df['right1000'] + df['right2000'] + df['right4000']) / 3
df['pta_avg'] = (df['pta_left_avg'] + df['pta_right_avg']) / 2
df['qscore'] = df['fafmc'] + df['tut'] + df['ffahs'] + df['hpyfs'] + df['mfaf'] + df['wyciyh']

# Filter values greater than 2 in specific columns
cols = ['fafmc', 'tut', 'ffahs', 'hpyfs', 'mfaf', 'wyciyh']
before_filter = len(df)
df = df[~(df[cols] > 2).any(axis=1)]
print(f"Dropped {before_filter - len(df)} rows with values > 2 in {cols} ({before_filter} → {len(df)})")

# Save dataset
model_input_file_path = r"C:\Users\sroes\PycharmProjects\PythonProject\model_input_thesis.csv"
df.to_csv(model_input_file_path, index=False)

# summary statistics
summary_stats = df.describe(include="all")
summary_file_path = r"C:\Users\sroes\OneDrive - ilionx Group BV\Bureaublad\Clementine\summary_statistics.csv"
summary_stats.to_csv(summary_file_path)

print(f"Final number of rows: {len(df)}")
