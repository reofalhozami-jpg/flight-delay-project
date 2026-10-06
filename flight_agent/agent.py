from google.adk.agents import Agent
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg://postgres:mypassword@localhost:5555/postgres')


def run_sql(query: str) -> dict:
    """Runs a read-only SQL SELECT query on the flight database and returns the rows.

    Args:
        query: A SQL SELECT query.
    """
    cleaned = query.strip().rstrip(';')
    if not cleaned.lower().startswith('select'):
        return {'error': 'Only SELECT queries are allowed.'}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(cleaned)).fetchmany(50)
            return {'rows': [list(r) for r in rows]}
    except Exception as e:
        return {'error': str(e)}


root_agent = Agent(
    name='flight_agent',
    model='gemini-3.5-flash-lite',
    description='Answers questions about US flights in January 2024 using the flight database.',
    instruction="""You are a data assistant for a flight database. Answer questions about US flights in January 2024 by running SQL queries with the run_sql tool.

The tables are:
- flights: the raw data, including cancelled flights
- flights_processed: cleaned data with cancelled flights removed
- flight_predictions and model_results also exist.

Column names are in UPPER CASE and must be written in double quotes in SQL, like "OP_UNIQUE_CARRIER". The main columns are FL_DATE, OP_UNIQUE_CARRIER (airline code), ORIGIN, DEST, DEP_DELAY, ARR_DELAY, ARR_DEL15 (1 if the flight arrived 15+ minutes late), CANCELLED (1 if cancelled), and DISTANCE.

Write simple SQL only: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT. Do not use GROUPING SETS, window functions, or CTEs.
Airline names are not in the database, only the two-letter codes, so give the code and say which airline it is from your own knowledge.
Never state a number unless it came from a query result. If you are unsure, run another query.
If a question can't be answered from this data, say so instead of guessing.""",
    tools=[run_sql],
)