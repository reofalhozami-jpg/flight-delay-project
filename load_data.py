import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("T_ONTIME_REPORTING.csv")

engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')

df.to_sql('flights', engine, if_exists='replace', index=False)

print("Data loaded successfully")