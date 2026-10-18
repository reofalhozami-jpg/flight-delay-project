# Copies the flights and flights_processed tables from the local database to the AWS database.
import os
import pandas as pd
from sqlalchemy import create_engine

local = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')
cloud = create_engine(os.environ['RDS_URL'])

for table in ['flights', 'flights_processed']:
    print('Copying', table)
    first = True
    for chunk in pd.read_sql(f'SELECT * FROM {table}', local, chunksize=50000):
        chunk.to_sql(table, cloud, if_exists='replace' if first else 'append', index=False)
        first = False
        print('  ', len(chunk), 'rows done')
print('Finished')