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

# --- Summary numbers for the top cards ---
total_flights = f"{len(df_c):,}"
delay_rate = round(df['ARR_DEL15'].mean() * 100, 1)
cancel_rate = round(df_c['CANCELLED'].mean() * 100, 1)
carrier_count = df_c['OP_UNIQUE_CARRIER'].nunique()

# --- Data for the charts (read from the database) ---
delay_by_airline = pd.read_sql(
    'SELECT "OP_UNIQUE_CARRIER" AS carrier, AVG("ARR_DEL15") * 100 AS rate '
    'FROM flights_processed GROUP BY "OP_UNIQUE_CARRIER" ORDER BY rate DESC', engine)

cancel_by_airport = pd.read_sql(
    'SELECT "ORIGIN" AS airport, COUNT(*) AS n FROM flights '
    'WHERE "CANCELLED" = 1 GROUP BY "ORIGIN" ORDER BY n DESC LIMIT 10', engine)

airline_labels = delay_by_airline['carrier'].tolist()
airline_values = delay_by_airline['rate'].round(2).tolist()
airport_labels = cancel_by_airport['airport'].tolist()
airport_values = cancel_by_airport['n'].tolist()


def make_row(columns, values, categories):
    row = pd.DataFrame([values])
    for col in columns:
        for prefix, chosen in categories.items():
            if col.startswith(prefix + '_'):
                row[col] = 1 if col == f"{prefix}_{chosen}" else 0
    return row.reindex(columns=columns, fill_value=0)


def chance_of_yes(model, row, base_rate):
    classes = list(model.classes_)
    p = min(model.predict_proba(row)[0][classes.index(1)], 0.999)
    odds = (p / (1 - p)) * (base_rate / (1 - base_rate))
    return odds / (1 + odds) * 100


# --- The web page ---
PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flight Delay Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://cdn.jsdelivr.net/npm/@n8n/chat/dist/style.css" rel="stylesheet" />
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f1f5f9; margin: 0; color: #0f172a; }

  .topbar { background: linear-gradient(135deg, #1e3a8a, #2563eb); color: white; padding: 28px 40px; }
  .topbar h1 { margin: 0; font-size: 26px; font-weight: 600; }
  .topbar p { margin: 6px 0 0 0; opacity: 0.85; font-size: 14px; }

  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 400px; gap: 24px; max-width: 1500px; margin: 0 auto; padding: 24px 40px 40px 40px; }
  .main { display: flex; flex-direction: column; gap: 24px; min-width: 0; }

  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .stat { background: white; border-radius: 12px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(15,23,42,0.08); border-left: 4px solid #2563eb; }
  .stat .label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat .value { font-size: 26px; font-weight: 600; margin-top: 4px; }
  .stat.red { border-left-color: #dc2626; }
  .stat.amber { border-left-color: #d97706; }
  .stat.green { border-left-color: #16a34a; }

  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 1px 3px rgba(15,23,42,0.08); }
  h2 { font-size: 16px; font-weight: 600; margin: 0 0 16px 0; }
  .chart-box { position: relative; height: 280px; }

  label { display: block; margin-top: 12px; font-size: 13px; color: #475569; }
  select, input { width: 100%; padding: 9px 10px; margin-top: 4px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 14px; background: white; }
  select:focus, input:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 14px; }
  button.predict { margin-top: 20px; width: 100%; padding: 11px; background: #2563eb; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; }
  button.predict:hover { background: #1d4ed8; }

  .result { margin-top: 14px; padding: 12px; border-radius: 8px; text-align: center; font-weight: 600; }
  .bad { background: #fee2e2; color: #b91c1c; }
  .good { background: #dcfce7; color: #15803d; }
  .error { background: #fef3c7; color: #92400e; }
  .note { font-size: 12px; color: #64748b; margin-top: 12px; line-height: 1.5; }

  .chat-panel { position: sticky; top: 24px; height: calc(100vh - 48px); display: flex; flex-direction: column; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(15,23,42,0.08); overflow: hidden; }
  .chat-title { padding: 18px 22px; border-bottom: 1px solid #e2e8f0; font-weight: 600; font-size: 16px; }
  .chat-title span { display: block; font-weight: 400; font-size: 12px; color: #64748b; margin-top: 2px; }
  #n8n-chat { flex: 1; min-height: 0; --chat--color-primary: #2563eb; --chat--color-primary-shade-50: #1d4ed8; --chat--color-primary-shade-100: #1e40af; --chat--border-radius: 10px; --chat--spacing: 0.9rem; }
  #n8n-chat .chat-layout { height: 100%; }
  #n8n-chat .chat-header { display: none; }

  @media (max-width: 1100px) {
    .layout { grid-template-columns: 1fr; padding: 20px; }
    .chat-panel { position: static; height: 600px; }
    .stats { grid-template-columns: 1fr 1fr; }
    .row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

  <div class="topbar">
    <h1>Flight Delay Dashboard</h1>
    <p>US domestic flights, January 2024</p>
  </div>

  <div class="layout">

    <div class="main">

      <div class="stats">
        <div class="stat"><div class="label">Total flights</div><div class="value">{{ total_flights }}</div></div>
        <div class="stat amber"><div class="label">Delay rate</div><div class="value">{{ delay_rate }}%</div></div>
        <div class="stat red"><div class="label">Cancellation rate</div><div class="value">{{ cancel_rate }}%</div></div>
        <div class="stat green"><div class="label">Airlines</div><div class="value">{{ carrier_count }}</div></div>
      </div>

      <div class="row">
        <div class="card">
          <h2>Predict a flight</h2>
          <form method="post">
            <div class="form-grid">
              <div>
                <label>Carrier</label>
                <select name="carrier">
                  {% for c in carriers %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
                </select>
              </div>
              <div>
                <label>Origin airport</label>
                <select name="origin">
                  {% for a in airports %}<option value="{{ a }}">{{ a }}</option>{% endfor %}
                </select>
              </div>
              <div>
                <label>Day of week (0=Mon, 6=Sun)</label>
                <input name="day_of_week" value="0">
              </div>
              <div>
                <label>Month</label>
                <input name="month" value="1">
              </div>
              <div>
                <label>Distance (miles)</label>
                <input name="distance" value="500">
              </div>
              <div>
                <label>Departure time (e.g. 1430)</label>
                <input name="dep_time" value="1200">
              </div>
            </div>
            <button class="predict" type="submit">Predict</button>
          </form>

          {% if error %}
            <div class="result error">{{ error }}</div>
          {% endif %}

          {% if result %}
            <div class="result {{ 'bad' if result.delayed else 'good' }}">
              {{ 'Higher risk of delay' if result.delayed else 'Normal delay risk' }} &mdash; chance of delay {{ result.delay_pct }}%
            </div>
            <div class="result {{ 'bad' if result.cancelled else 'good' }}">
              {{ 'Higher risk of cancellation' if result.cancelled else 'Normal cancellation risk' }} &mdash; chance of cancellation {{ result.cancel_pct }}%
            </div>
          {% endif %}
        </div>

        <div class="card">
          <h2>Delay rate by airline (%)</h2>
          <div class="chart-box"><canvas id="airlineChart"></canvas></div>
        </div>
      </div>

      <div class="card">
        <h2>Top 10 airports by cancelled flights</h2>
        <div class="chart-box"><canvas id="airportChart"></canvas></div>
      </div>

    </div>

    <div class="chat-panel">
      <div class="chat-title">Ask about the flight data<span>Answers come straight from the database</span></div>
      <div id="n8n-chat"></div>
    </div>

  </div>

<script>
  new Chart(document.getElementById('airlineChart'), {
    type: 'bar',
    data: {
      labels: {{ airline_labels | tojson }},
      datasets: [{ label: 'Delay rate (%)', data: {{ airline_values | tojson }}, backgroundColor: '#2563eb', borderRadius: 4 }]
    },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } } }
  });

  new Chart(document.getElementById('airportChart'), {
    type: 'bar',
    data: {
      labels: {{ airport_labels | tojson }},
      datasets: [{ label: 'Cancelled flights', data: {{ airport_values | tojson }}, backgroundColor: '#dc2626', borderRadius: 4 }]
    },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } } }
  });
</script>

<script type="module">
  import { createChat } from 'https://cdn.jsdelivr.net/npm/@n8n/chat/dist/chat.bundle.es.js';
  createChat({
    webhookUrl: 'http://localhost:5678/webhook/9de7b9f0-2472-40e6-b861-ed75c109126b/chat',
    target: '#n8n-chat',
    mode: 'fullscreen',
    showWelcomeScreen: false,
    initialMessages: ['Ask me anything about the January 2024 flight data.'],
    i18n: {
      en: {
        title: '',
        subtitle: '',
        footer: '',
        getStarted: 'New conversation',
        inputPlaceholder: 'Ask a question about the flights...'
      }
    }
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
                'delay_pct': round(chance_of_yes(delay_model, delay_row, delay_rate / 100), 1),
                'cancelled': cancel_model.predict(cancel_row)[0] == 1,
                'cancel_pct': round(chance_of_yes(cancel_model, cancel_row, cancel_rate / 100), 1),
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
        total_flights=total_flights,
        delay_rate=delay_rate,
        cancel_rate=cancel_rate,
        carrier_count=carrier_count,
    )


if __name__ == "__main__":
    app.run(debug=False)