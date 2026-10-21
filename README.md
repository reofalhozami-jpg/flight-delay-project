# Flight Delay Prediction

Predicts whether a single US flight will be delayed or cancelled, using real January 2024 flight data. The project has a database, a preprocessing and modeling pipeline, a web dashboard, and two chatbots (n8n and Google ADK) that answer questions about the data by writing SQL.

## What it does

- Takes a flight (airline, origin airport, day of week, distance, scheduled departure time) and returns the chance of a delay and the chance of a cancellation.
- Shows summary numbers and two charts (delay rate by airline, airports with the most cancellations).
- Has a chat box where you can ask questions about the data. The chatbot writes a SQL query, runs it on the database, and explains the result.

## Data

- Source: US Bureau of Transportation Statistics, On-Time Reporting (transtats.bts.gov), January 2024.
- File: `T_ONTIME_REPORTING.csv`, 547,271 flights, 18 columns. It is not in the repository because of its size. Download it from the BTS site and put it in the project folder.

Findings from the EDA (`eda_flights.ipynb`):

- 3.7% of flights were cancelled (20,389 of 547,271).
- About 24% of flights arrived 15 or more minutes late.
- AA and B6 had the highest delay rates (about 29%). YX and DL had the lowest.
- Tuesday and Friday had the most delays. Wednesday and Thursday had the fewest.
- Cancelled flights have blank departure and arrival fields. This is expected, not a data error.

## Database

PostgreSQL, running in Docker.

| Table                | Rows           | Content                                                                    |
| -------------------- | -------------- | -------------------------------------------------------------------------- |
| `flights`            | 547,271        | Raw data, loaded from the CSV, including cancelled flights                 |
| `flights_processed`  | 526,882        | Cleaned data for the delay model                                           |
| `flight_predictions` | 105,377        | Delay model predictions on the test set, next to the real result           |
| `model_results`      | grows each run | Accuracy, precision, recall and F1 for each model, logged by `evaluate.py` |

`schema.sql` describes the `flights` table. The real table uses the original uppercase column names from the CSV.

## Pipeline and files

| File                           | What it does                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `load_data.py`                 | Loads the CSV into the `flights` table                                                                                          |
| `preprocess.py`                | Removes cancelled flights, adds `day_of_week` and `month`, drops unused columns, fills blanks with 0, saves `flights_processed` |
| `train_model.py`               | Baseline model (Logistic Regression)                                                                                            |
| `train_model_improved.py`      | Random Forest models, including the balanced one                                                                                |
| `train_cancellation_model.py`  | Cancellation model                                                                                                              |
| `evaluate.py`                  | Trains all models, computes metrics, logs them to `model_results`                                                               |
| `postprocess.py`               | Converts 0 and 1 into "On Time" and "Delayed"                                                                                   |
| `save_results.py`              | Saves test predictions to `flight_predictions`                                                                                  |
| `app.py`                       | Flask web app: prediction form, charts, chat box                                                                                |
| `flight_agent/agent.py`        | Google ADK chatbot with a read-only SQL tool                                                                                    |
| `n8n_flight_chatbot.json`      | Exported n8n chatbot workflow                                                                                                   |
| `Dockerfile`, `Dockerfile.adk` | Container images for the website and the ADK chatbot                                                                            |
| `docker-compose.yml`           | Starts the website, the ADK chatbot and (optionally) the database with one command                                              |
| `copy_to_rds.py`               | Copies the tables from the local database to the cloud database                                                                 |

## Models and decisions

Delay model target: `ARR_DEL15` (1 if the flight arrived 15 or more minutes late). Cancellation model target: `CANCELLED`. Features: airline, day of week, month, distance, scheduled departure time (the cancellation model also uses the origin airport). Data is split 80% training, 20% test, randomly, with `random_state=42`.

| Model                                  | Accuracy | Precision | Recall | F1     |
| -------------------------------------- | -------- | --------- | ------ | ------ |
| Baseline, Logistic Regression (delay)  | 76.13%   | 0%        | 0%     | 0%     |
| Random Forest, unbalanced (delay)      | 76.13%   | 0%        | 0%     | 0%     |
| Random Forest, balanced (delay)        | 57.24%   | 30.75%    | 63.22% | 41.37% |
| Random Forest, balanced (cancellation) | 73.95%   | 8.65%     | 63.66% | 15.23% |

Key decision: the baseline reached 76% accuracy by predicting "on time" for every flight, since about 76% of flights are on time. It found none of the real delays. I switched to Random Forest with `class_weight='balanced'`, which finds about 63% of real delays. Accuracy drops because the model raises more false alarms. For this problem, finding delays matters more than overall accuracy.

The balanced weighting pushes the raw model probabilities far above the real rates. `app.py` rescales them to the real delay and cancellation rates before showing a percentage, so the website shows a realistic chance.

Why Random Forest: it works on mixed numeric and category data with little tuning, supports class weights, gives probabilities, and is easier to explain than boosting or neural networks.

## Run locally

Requirements: Python 3, Docker Desktop.

1. Create the network and start the database:

```
docker network create flight-net
docker run -d --name flight-postgres --network flight-net -e POSTGRES_PASSWORD=<DB_PASSWORD> -p 5555:5432 postgres
```

2. Install the packages and build the database:

```
pip install -r requirements.txt
python load_data.py
python preprocess.py
python evaluate.py
python save_results.py
```

3. Run the website:

```
python app.py
```

Open http://127.0.0.1:5000. The chat box needs the ADK chatbot running (step 4).

4. Run the ADK chatbot. Create `flight_agent/.env` with `GOOGLE_API_KEY=<your key>` and `GOOGLE_GENAI_USE_VERTEXAI=FALSE`, then:

```
adk web
```

Open http://127.0.0.1:8000 and pick `flight_agent`. To use it in the website chat box, run it on port 8000 before starting `app.py`.

5. Optional, n8n chatbot:

```
docker run -d --name n8n --network flight-net -p 5678:5678 docker.n8n.io/n8nio/n8n
```

Open http://localhost:5678, import `n8n_flight_chatbot.json`, and add your Gemini key and a Postgres connection (host `flight-postgres`, port 5432) in the credentials.

## Run in Docker

```
docker build -t flight-app .
docker run -d --name flight-app --network flight-net -p 5000:5000 -e DATABASE_URL=postgresql://postgres:<DB_PASSWORD>@flight-postgres:5432/postgres flight-app
```

The app reads the database address from `DATABASE_URL` and falls back to the local address.

## Run with Docker Compose

`docker-compose.yml` starts the website and the ADK chatbot together. The website talks to the chatbot at `http://adk:8000` inside the Docker network.

1. Create the network, and put your key in `flight_agent/.env` (see step 4 above):

```
docker network create flight-net
```

2. Start the database. If you already have a `flight-postgres` container from the local steps, skip this and use it. Otherwise, start it from the compose file and load the data (steps 2 of "Run locally"):

```
docker compose --profile db up -d db
```

3. Start the website and the chatbot:

```
docker compose up -d --build
```

Open http://127.0.0.1:5000. The website needs one to two minutes to train its models. To stop everything: `docker compose down`.

Set `DB_PASSWORD` in your shell to use a password other than the default.

## Run in the cloud (AWS, region eu-north-1)

The website runs on ECS Fargate behind an Application Load Balancer. The database is Amazon RDS for PostgreSQL. The website and the ADK chatbot run as two containers in the same task, so the chatbot is never exposed to the internet.

**How to use it:** open the load balancer's DNS name in a browser (plain `http`). The website's chat box talks to the ADK chatbot inside the task, so the chatbot is used through the website and has no public address of its own.

1. Push the images: create ECR repositories `flight-app` and `flight-adk`, log Docker in with `aws ecr get-login-password`, then `docker tag` and `docker push` both images.
2. Create the database: `aws rds create-db-instance` with PostgreSQL (db.t4g.micro, 20 GB), then copy the data with `copy_to_rds.py`:

```
RDS_URL=postgresql://postgres:<DB_PASSWORD>@<RDS_ENDPOINT>:5432/postgres?sslmode=require python copy_to_rds.py
```

3. Network: security group for the load balancer (port 80 from anywhere), a security group for the containers (port 5000 from the load balancer only), and a database rule that allows port 5432 from the container group.
4. Store the Gemini key in AWS Systems Manager Parameter Store as a SecureString, and let the task execution role read it.
5. Create an ECS cluster, a log group, and a task definition with two containers (`flight-app` on port 5000 and `flight-adk` on port 8000). The task definition holds the database URL, so it is not committed.
6. Create the load balancer, a target group (port 5000, health check on `/`), and a listener on port 80.
7. Create the ECS service with 2 tasks and a 300 second health check grace period, because the app takes one to two minutes to train its models on start.
8. Auto scaling: minimum 2, maximum 4 tasks, scaling when average CPU passes 60%.

## Limitations

- Only one month of data, so the models cannot learn seasonal patterns and the form has no month field.
- Blank values are filled with 0, which is a simple fix. A blank does not really mean zero.
- The balanced models trade accuracy for recall. Precision is low (31% for delays, 9% for cancellations), so most flagged flights are false alarms.
- The percentages are estimates rescaled to the real rates, not guarantees. The cancellation percentage in particular can run well above the real rate for airlines that rarely cancel.
- The models use five or six features and ignore weather, aircraft and airport congestion.
- The train and test split is random, not stratified, and not split by date.
- The app trains both models every time it starts, which takes one to two minutes.
- The chatbot is restricted to SELECT queries by a simple check and by its instructions. A database user with read-only rights would be safer.
- The chatbot's numbers come from SQL, but names it adds from its own knowledge (for example airport names) can be wrong.
- Questions like "worst airport" are ambiguous, and the chatbot's answer depends on the cutoff it picks.
- The cloud site uses plain HTTP and has no login, and anyone with the link can use the chat box, which uses the Gemini key. The database password is passed as an environment variable. A production setup would use Secrets Manager and HTTPS.
- n8n runs only on the local machine. The cloud version uses the ADK chatbot.
- The Docker Compose file was tested with an existing `flight-postgres` container. Starting the database from the compose file's `db` service on a clean machine has not been tested.
