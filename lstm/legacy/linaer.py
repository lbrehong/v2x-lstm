import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Read the CSV file
filename = input('Enter the file path: ')
# Define the threshold for outliers
threshold = float(input('Enter the threshold value for the outliers: '))
df = pd.read_csv(filename)

# Filter out the outliers
filtered_df = df[df['latency_ms'] <= threshold]

# Prepare the data for linear regression
X = filtered_df[['tx_seq_num']].values.reshape(-1, 1)
y = filtered_df['latency_ms'].values

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Get the coefficient
coefficient = model.coef_[0]

print(f"The coefficient is: {coefficient}")
