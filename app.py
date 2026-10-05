from flask import Flask, request, render_template_string
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')

# --- Delay model (trained once, when the app starts) ---
df = pd.read_sql('SELECT * FROM flights_processed', engine)

delay_features = ['OP_UNIQUE_CARRIER', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
X = pd.get_dummies(df[delay_features], columns=['OP_UNIQUE_CARRIER'])
y = df['ARR_DEL15']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

delay_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
delay_model.fit(X_train, y_train)
delay_columns = X.columns

carrier_list = sorted(df['OP_UNIQUE_CARRIER'].unique())

# --- Cancellation model (trained once, when the app starts) ---
df_c = pd.read_sql('SELECT * FROM flights', engine)
df_c['FL_DATE'] = pd.to_datetime(df_c['FL_DATE'])
df_c['day_of_week'] = df_c['FL_DATE'].dt.dayofweek
df_c['month'] = df_c['FL_DATE'].dt.month

cancel_features = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'day_of_week', 'month', 'DISTANCE', 'CRS_DEP_TIME']
Xc = pd.get_dummies(df_c[cancel_features], columns=['OP_UNIQUE_CARRIER', 'ORIGIN'])
yc = df_c['CANCELLED']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

cancel_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
cancel_model.fit(Xc_train, yc_train)
cancel_columns = Xc.columns

airport_list = sorted(df_c['ORIGIN'].unique())

# --- Data for the charts and the table (read from the database) ---
delay_by_airline = pd.read_sql(
    'SELECT "OP_UNIQUE_CARRIER" AS carrier, AVG("ARR_DEL15") * 100 AS rate '
    'FROM flights_processed GROUP BY "OP_UNIQUE_CARRIER" ORDER BY rate DESC', engine)

cancel_by_airport = pd.read_sql(
    'SELECT "ORIGIN" AS airport, COUNT(*) AS n FROM flights '
    'WHERE "CANCELLED" = 1 GROUP BY "ORIGIN" ORDER BY n DESC LIMIT 10', engine)

model_table = pd.read_sql(
    'SELECT DISTINCT ON (model) model, accuracy, "precision", recall, f1 '
    'FROM model_results ORDER BY model, run_time DESC', engine)
for col in ['accuracy', 'precision', 'recall', 'f1']:
    model_table[col] = (model_table[col] * 100).round(2)

airline_labels = delay_by_airline['carrier'].tolist()
airline_values = delay_by_airline['rate'].round(2).tolist()
airport_labels = cancel_by_airport['airport'].tolist()
airport_values = cancel_by_airport['n'].tolist()
model_rows = model_table.to_dict('records')


def make_row(columns, values, categories):
    row = pd.DataFrame([values])
    for col in columns:
        for prefix, chosen in categories.items():
            if col.startswith(prefix + '_'):
                row[col] = 1 if col == f"{prefix}_{chosen}" else 0
    return row.reindex(columns=columns, fill_value=0)


def chance_of_yes(model, row):
    classes = list(model.classes_)
    return model.predict_proba(row)[0][classes.index(1)] * 100


# --- The web page ---
PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Flight Delay Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 30px; }
  h1 { text-align: center; color: #222; margin-bottom: 5px; }
  .sub { text-align: center; color: #666; margin-bottom: 30px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1100px; margin: 0 auto; }
  .card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
  .wide { grid-column: 1 / 3; }
  h2 { font-size: 18px; color: #222; margin-top: 0; }
  label { display: block; margin-top: 12px; font-size: 14px; color: #444; }
  select, input { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }
  button { margin-top: 20px; width: 100%; padding: 10px; background: #2563eb; color: white; border: none; border-radius: 5px; font-size: 15px; cursor: pointer; }
  button:hover { background: #1e4fc4; }
  .result { margin-top: 15px; padding: 12px; border-radius: 6px; text-align: center; font-weight: bold; }
  .bad { background: #fde2e2; color: #b91c1c; }
  .good { background: #dcfce7; color: #15803d; }
  .error { background: #fef3c7; color: #92400e; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }
  th { background: #f4f6f8; }
  .note { font-size: 13px; color: #666; margin-top: 10px; }
</style>
</head>
<body>
  <h1>Flight Delay Dashboard</h1>
  <div class="sub">US flights, January 2024</div>

  <div class="grid">

    <div class="card">
      <h2>Predict a flight</h2>
      <form method="post">
        <label>Carrier</label>
        <select name="carrier">
          {% for c in carriers %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
        </select>

        <label>Origin airport</label>
        <select name="origin">
          {% for a in airports %}<option value="{{ a }}">{{ a }}</option>{% endfor %}
        </select>

        <label>Day of week (0=Monday, 6=Sunday)</label>
        <input name="day_of_week" value="0">

        <label>Month</label>
        <input name="month" value="1">

        <label>Distance (miles)</label>
        <input name="distance" value="500">

        <label>Scheduled departure time (e.g. 1430)</label>
        <input name="dep_time" value="1200">

        <button type="submit">Predict</button>
      </form>

      {% if error %}
        <div class="result error">{{ error }}</div>
      {% endif %}

      {% if result %}
        <div class="result {{ 'bad' if result.delayed else 'good' }}">
          {{ 'Delayed' if result.delayed else 'On Time' }} &mdash; delay score {{ result.delay_pct }}%
        </div>
        <div class="result {{ 'bad' if result.cancelled else 'good' }}">
          {{ 'Cancelled' if result.cancelled else 'Not Cancelled' }} &mdash; cancellation score {{ result.cancel_pct }}%
        </div>
        <div class="note">
          These scores come from models set up to catch more problem flights, so they run higher than the real
          rates (about 24% of flights are delayed and about 4% are cancelled).
        </div>
      {% endif %}
    </div>

    <div class="card">
      <h2>Delay rate by airline (%)</h2>
      <canvas id="airlineChart"></canvas>
    </div>

    <div class="card wide">
      <h2>Top 10 airports by cancelled flights</h2>
      <canvas id="airportChart" height="90"></canvas>
    </div>

    <div class="card wide">
      <h2>Model comparison (%)</h2>
      <table>
        <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        {% for m in models %}
        <tr>
          <td>{{ m['model'] }}</td>
          <td>{{ m['accuracy'] }}</td>
          <td>{{ m['precision'] }}</td>
          <td>{{ m['recall'] }}</td>
          <td>{{ m['f1'] }}</td>
        </tr>
        {% endfor %}
      </table>
      <div class="note">The form uses the balanced Random Forest for delays and the balanced Random Forest for cancellations.</div>
    </div>

  </div>

<script>
  new Chart(document.getElementById('airlineChart'), {
    type: 'bar',
    data: {
      labels: {{ airline_labels | tojson }},
      datasets: [{ label: 'Delay rate (%)', data: {{ airline_values | tojson }}, backgroundColor: '#2563eb' }]
    },
    options: { plugins: { legend: { display: false } } }
  });

  new Chart(document.getElementById('airportChart'), {
    type: 'bar',
    data: {
      labels: {{ airport_labels | tojson }},
      datasets: [{ label: 'Cancelled flights', data: {{ airport_values | tojson }}, backgroundColor: '#dc2626' }]
    },
    options: { plugins: { legend: { display: false } } }
  });
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    if request.method == "POST":
        try:
            day = int(request.form['day_of_week'])
            month = int(request.form['month'])
            distance = float(request.form['distance'])
            dep_time = int(request.form['dep_time'])
            carrier = request.form['carrier']
            origin = request.form['origin']

            delay_row = make_row(
                delay_columns,
                {'day_of_week': day, 'month': month, 'DISTANCE': distance, 'CRS_DEP_TIME': dep_time},
                {'OP_UNIQUE_CARRIER': carrier},
            )
            cancel_row = make_row(
                cancel_columns,
                {'day_of_week': day, 'month': month, 'DISTANCE': distance, 'CRS_DEP_TIME': dep_time},
                {'OP_UNIQUE_CARRIER': carrier, 'ORIGIN': origin},
            )

            result = {
                'delayed': delay_model.predict(delay_row)[0] == 1,
                'delay_pct': round(chance_of_yes(delay_model, delay_row), 1),
                'cancelled': cancel_model.predict(cancel_row)[0] == 1,
                'cancel_pct': round(chance_of_yes(cancel_model, cancel_row), 1),
            }
        except Exception as e:
            error = f"Error: {e}"

    return render_template_string(
        PAGE,
        carriers=carrier_list,
        airports=airport_list,
        result=result,
        error=error,
        airline_labels=airline_labels,
        airline_values=airline_values,
        airport_labels=airport_labels,
        airport_values=airport_values,
        models=model_rows,
    )


if __name__ == "__main__":
    app.run(debug=False)