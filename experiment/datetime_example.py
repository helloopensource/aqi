import pandas as pd
from datetime import datetime

# Example datetime string
datetime_str = '2022-01-01T07:00:00Z'

# Parse the datetime string
dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%SZ')

# Create DataFrame with scalar values - Method 1: Using index
date_df = pd.DataFrame({
    'day': [dt.date()],  # Wrap in list to make it a series
    'avg': [42.5]        # Example average value
})

print("\nMethod 1 - Using list for scalar values:")
print(date_df)

# Method 2: Using index parameter
date_df2 = pd.DataFrame({
    'day': dt.date(),
    'avg': 42.5
}, index=[0])  # Specify an index

print("\nMethod 2 - Using index parameter:")
print(date_df2)

# Method 3: Using pd.Series
date_df3 = pd.DataFrame({
    'day': pd.Series(dt.date(), index=[0]),
    'avg': pd.Series(42.5, index=[0])
})

print("\nMethod 3 - Using pd.Series:")
print(date_df3)

# Example of concatenating multiple rows
dates = [
    '2022-01-01T07:00:00Z',
    '2022-01-02T08:30:00Z',
    '2022-01-03T09:45:00Z'
]

# Create a list of DataFrames
dfs = []
for date_str in dates:
    dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
    df = pd.DataFrame({
        'day': [dt.date()],
        'avg': [42.5]  # Example average value
    })
    dfs.append(df)

# Concatenate all DataFrames
final_df = pd.concat(dfs, ignore_index=True)
print("\nConcatenated DataFrame:")
print(final_df) 