import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('weather_data.csv')

# Preprocess the dataset
df['wind_bearing'] = df['wind_bearing'].astype(float)
df['uv_index'] = df['uv_index'].astype(float)

X = df.drop(["ummary", "time", "icon"], axis=1)
y = df["ummary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train the RandomForest model
rf = RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf, 'random_forest_model.pkl')
