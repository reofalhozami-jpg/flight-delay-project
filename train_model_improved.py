# Baseline (Logistic Regression): 76.13% accuracy
# Improved (Random Forest): 76.13% accuracy
# Both models landed on the same result, roughly matching the rate you'd
# get by always predicting "not delayed." This suggests the current
# features (carrier, day, month, distance, departure time) aren't
# sufficient to meaningfully predict delays - weather or operational
# data would likely be needed for real improvement.

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect to the database
engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')

# Load the cleaned data
df = pd.read_sql('SELECT * FROM flights_processed', engine)

# Same features as before
features = ['OP_UNIQUE_CARRIER', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
target = 'ARR_DEL15'

X = df[features]
y = df[target]

X = pd.get_dummies(X, columns=['OP_UNIQUE_CARRIER'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Improved model: Random Forest instead of Logistic Regression
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Improved model (Random Forest) accuracy: {accuracy:.2%}")