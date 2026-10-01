import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')

# Use the raw flights table becausit has cancelled flights
df = pd.read_sql('SELECT * FROM flights', engine)

# Build the same day_of_week / month features as before
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['day_of_week'] = df['FL_DATE'].dt.dayofweek
df['month'] = df['FL_DATE'].dt.month

features = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
target = 'CANCELLED'

X = df[features]
y = df[target]

X = pd.get_dummies(X, columns=['OP_UNIQUE_CARRIER', 'ORIGIN'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Cancellation model")
print(f"  accuracy:  {accuracy_score(y_test, preds):.2%}")
print(f"  precision: {precision_score(y_test, preds, zero_division=0):.2%}")
print(f"  recall:    {recall_score(y_test, preds, zero_division=0):.2%}")
print(f"  f1:        {f1_score(y_test, preds, zero_division=0):.2%}")

always_not_cancelled = 1 - y_test.mean()
print(f"\nAlways guessing 'Not Cancelled' would score: {always_not_cancelled:.2%}")