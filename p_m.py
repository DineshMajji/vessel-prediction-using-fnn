import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Load the data
data = pd.read_csv("/content/data.csv")
data = data.drop("index", axis=1)

# Encode categorical data
enc = LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = enc.fit_transform(data[i])

data.columns=['Lever position ', 'Ship speed (v) ',
       'Gas Turbine (GT) shaft torque (GTT) [kN m]  ',
       'GT rate of revolutions (GTn) [rpm]  ',
       'Gas Generator rate of revolutions (GGn) [rpm]  ',
       'Starboard Propeller Torque (Ts) [kN]  ',
       'Port Propeller Torque (Tp) [kN]  ',
       'Hight Pressure (HP) Turbine exit temperature (T48) [C]  ',
       'GT Compressor inlet air temperature (T1) [C]  ',
       'GT Compressor outlet air temperature (T2) [C]  ',
       'HP Turbine exit pressure (P48) [bar]  ',
       'GT Compressor inlet air pressure (P1) [bar]  ',
       'GT Compressor outlet air pressure (P2) [bar]  ',
       'GT exhaust gas pressure (Pexh) [bar]  ',
       'Turbine Injecton Control (TIC) [%]  ', 'Fuel flow (mf) [kg/s]  ','compressor decay','turbine decay']

y = data[['compressor decay', 'turbine decay']]
x = data.drop(['compressor decay', 'turbine decay'], axis=1)

# Standardize and normalize input features
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = Normalizer().fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Train Decision Tree Regressor
model1 = DecisionTreeRegressor(random_state=0)
model1 = model1.fit(x_train, y_train)

# Train Random Forest Regressor
model2 = RandomForestRegressor(n_estimators=300, max_depth=25)
model2.fit(x_train, y_train)

# Take user input for testing
user_input = {}
for col in data.columns:
    if col != 'compressor decay' and col != 'turbine decay':
        user_input[col] = float(input(f"Enter value for {col}: "))

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Standardize and normalize user input
user_input_scaled = scaler.transform(user_df)
user_input_normalized = Normalizer().transform(user_input_scaled)

# Predict using both models on user input
p1 = model1.predict(user_input_normalized)
p2 = model2.predict(user_input_normalized)

# Calculate mean values of predictions
mean_p1 = np.mean(p1)
mean_p2 = np.mean(p2)

# Check if both mean predictions are above 0.80
if mean_p1 > 0.80 and mean_p2 > 0.80:
    print("Perfect")
else:
    print("Unsafe")
