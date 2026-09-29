from flask import Flask, request, render_template_string
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- Train the model once, when the app starts ---
engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')
df = pd.read_sql('SELECT * FROM flights_processed', engine)

features = ['OP_UNIQUE_CARRIER', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
target = 'ARR_DEL15'

X = pd.get_dummies(df[features], columns=['OP_UNIQUE_CARRIER'])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

carrier_list = sorted(df['OP_UNIQUE_CARRIER'].unique())
model_columns = X.columns

# --- The web page ---
PAGE = """
<h1>Flight Delay Predictor</h1>
<form method="post">
  Carrier:
  <select name="carrier">
    {% for c in carriers %}<option value="{{c}}">{{c}}</option>{% endfor %}
  </select><br><br>
  Day of week (0=Monday, 6=Sunday): <input name="day_of_week" value="0"><br><br>
  Month: <input name="month" value="1"><br><br>
  Distance (miles): <input name="distance" value="500"><br><br>
  Scheduled departure time (e.g. 1430): <input name="dep_time" value="1200"><br><br>
  <button type="submit">Predict</button>
</form>
{% if result %}<h2>Prediction: {{ result }}</h2>{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            row = pd.DataFrame([{
                'day_of_week': int(request.form['day_of_week']),
                'month': int(request.form['month']),
                'DISTANCE': float(request.form['distance']),
                'CRS_DEP_TIME': int(request.form['dep_time']),
            }])
            for col in model_columns:
                if col.startswith('OP_UNIQUE_CARRIER_'):
                    row[col] = 1 if col == f"OP_UNIQUE_CARRIER_{request.form['carrier']}" else 0
            row = row.reindex(columns=model_columns, fill_value=0)

            prediction = model.predict(row)[0]
            result = "Delayed" if prediction == 1 else "On Time"
        except Exception as e:
            result = f"Error: {e}"

    return render_template_string(PAGE, carriers=carrier_list, result=result)

if __name__ == "__main__":
    app.run(debug=True)