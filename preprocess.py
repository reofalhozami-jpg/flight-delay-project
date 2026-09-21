import pandas as pd
from sqlalchemy import create_engine
# specifically for delay prediction
# Connect to the database
engine = create_engine('postgresql://postgres:mypassword@localhost:5555/postgres')

# Load the raw data from the flights table
df = pd.read_sql('SELECT * FROM flights', engine)

# --- Preprocessing steps ---

# 1. Cancelled flights have no delay info - that's expected, not an error.
#    For modeling, we'll focus on flights that actually flew.
df_clean = df[df['CANCELLED'] == 0].copy()

# 2. Convert fl_date to a real date object instead of a string
df_clean['FL_DATE'] = pd.to_datetime(df_clean['FL_DATE'])
df_clean['day_of_week'] = df_clean['FL_DATE'].dt.dayofweek # 0=Monday
df_clean['month'] = df_clean['FL_DATE'].dt.month #for later

# 3. Drop columns that wont help predict delays, wewill use 
df_clean = df_clean.drop(columns=['ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'CANCELLATION_CODE'])

# 4. Fill any remaining small gaps in numeric columns with 0
df_clean = df_clean.fillna(0)

# Save the cleaned data into a new table, keeping raw data untouched
df_clean.to_sql('flights_processed', engine, if_exists='replace', index=False)

print(f"Preprocessing done. {len(df_clean)} rows saved to flights_processed.")