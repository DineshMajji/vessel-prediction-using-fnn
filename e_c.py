import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
data = pd.read_excel("/content/energy2 -.xlsx")
print(data.columns)
data = data.drop("index", axis=1)
dataplot=sns.heatmap(data.corr())
plt.show()

data.columns = ['Vessel Speed in Kmph ', 'Engine_RPM ',
                'Water Flow in Kmph ',
                'Baseline_Energy ',
                'Air Flow  ',
                'Cargo Load in Kgs ',
                'Temperature in C  ',
                'Operational_Mode ',
                'Propulsion system efficiency  ',
                'Excess energy consumed ']

# Define target variable and input features
y = data['Excess energy consumed ']
x = data.drop('Excess energy consumed ', axis=1)

# Standardize input features
scaler = StandardScaler()
scaled_column_names = x.columns.tolist()

# Build the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # One output neuron for excess energy consumption
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape'])

# Train the model on the entire dataset
model.fit(x, y, epochs=100, batch_size=32)

# Take user input for testing
user_input = {}
for col in scaled_column_names:
    user_input[col] = float(input(f"Enter value for {col}: "))

# Convert user input to numpy array and standardize
scaler.fit(x)
user_input_array = np.array([list(user_input.values())])
user_input_array = scaler.transform(user_input_array)

# Make prediction
prediction = model.predict(user_input_array)

# Calculate excess energy consumed and print the result
excess_energy_consumed = prediction[0][0]+500
print(f"Excess energy consumed is {excess_energy_consumed:.2f}")
