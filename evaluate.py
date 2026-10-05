import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data (with basic error handling)
try:
    engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')
    df = pd.read_sql('SELECT * FROM flights_processed', engine)
except Exception as e:
    print("Could not load data. Is the Docker container running?")
    print(e)
    raise SystemExit(1)

features = ['OP_UNIQUE_CARRIER', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
target = 'ARR_DEL15'

X = pd.get_dummies(df[features], columns=['OP_UNIQUE_CARRIER'])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Baseline (Logistic Regression)': LogisticRegression(max_iter=1000),
    'Improved (Random Forest)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Improved v2 (Random Forest, balanced)': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
}

# Accuracy you'd get by always guessing "On Time"
always_on_time = 1 - y_test.mean()

log_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    row = {
        'run_time': datetime.now(),
        'model': name,
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'always_on_time_accuracy': always_on_time,
    }
    log_rows.append(row)
    print(name)
    print(f"  accuracy:  {row['accuracy']:.2%}")
    print(f"  precision: {row['precision']:.2%}")
    print(f"  recall:    {row['recall']:.2%}")
    print(f"  f1:        {row['f1']:.2%}")

print(f"\nAlways guessing 'On Time' would score: {always_on_time:.2%}")

# --- Cancellation model, using the raw flights table ---
df_cancel = pd.read_sql('SELECT * FROM flights', engine)
df_cancel['FL_DATE'] = pd.to_datetime(df_cancel['FL_DATE'])
df_cancel['day_of_week'] = df_cancel['FL_DATE'].dt.dayofweek
df_cancel['month'] = df_cancel['FL_DATE'].dt.month

cancel_features = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
Xc = pd.get_dummies(df_cancel[cancel_features], columns=['OP_UNIQUE_CARRIER', 'ORIGIN'])
yc = df_cancel['CANCELLED']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

cancel_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
cancel_model.fit(Xc_train, yc_train)
cancel_preds = cancel_model.predict(Xc_test)

cancel_row = {
    'run_time': datetime.now(),
    'model': 'Cancellation (Random Forest, balanced)',
    'accuracy': accuracy_score(yc_test, cancel_preds),
    'precision': precision_score(yc_test, cancel_preds, zero_division=0),
    'recall': recall_score(yc_test, cancel_preds, zero_division=0),
    'f1': f1_score(yc_test, cancel_preds, zero_division=0),
    'always_on_time_accuracy': 1 - yc_test.mean(),
}
log_rows.append(cancel_row)

print('Cancellation (Random Forest, balanced)')
print(f"  accuracy:  {cancel_row['accuracy']:.2%}")
print(f"  precision: {cancel_row['precision']:.2%}")
print(f"  recall:    {cancel_row['recall']:.2%}")
print(f"  f1:        {cancel_row['f1']:.2%}")

# Log the results into their own table
try:
    pd.DataFrame(log_rows).to_sql('model_results', engine, if_exists='append', index=False)
    print("Results saved to model_results table.")
except Exception as e:
    print("Could not save results to the database.")
    print(e)