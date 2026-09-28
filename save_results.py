import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')
df = pd.read_sql('SELECT * FROM flights_processed', engine)

features = ['OP_UNIQUE_CARRIER', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
target = 'ARR_DEL15'

X = df[features]
y = df[target]
X = pd.get_dummies(X, columns=['OP_UNIQUE_CARRIER'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# --- Build a results table ---
results = X_test.copy()
results['actual_delayed'] = y_test.values
results['predicted_delayed'] = predictions
results['prediction_label'] = results['predicted_delayed'].map({0: 'On Time', 1: 'Delayed'})

# --- Save predictions into their own dedicated table ---
results.to_sql('flight_predictions', engine, if_exists='replace', index=False)

print(f"Saved {len(results)} predictions to flight_predictions table.")