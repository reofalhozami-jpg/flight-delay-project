import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Connect to the database
engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')

# Load the cleaned data
df = pd.read_sql('SELECT * FROM flights_processed', engine)

# Pick simple features to start with
features = ['OP_UNIQUE_CARRIER', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
target = 'ARR_DEL15'

X = df[features]
y = df[target]

# Convert the carrier (airline) column into numbers a model can use
X = pd.get_dummies(X, columns=['OP_UNIQUE_CARRIER'])

# Split into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test how well it did
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Baseline model accuracy: {accuracy:.2%}")