-- Database schema for the flight delay project

-- Note: the tables below are created by the Python scripts (load_data.py,
-- preprocess.py, save_results.py, evaluate.py), not by running this file
-- This file documents their structure


-- 1: Raw data, loaded from T_ONTIME_REPORTING.csv by load_data.py
--    Includes cancelled flights. 547,271 rows.
CREATE TABLE flights (
    id SERIAL PRIMARY KEY,
    fl_date DATE,
    carrier VARCHAR(10),
    flight_num INTEGER,
    origin VARCHAR(10),
    origin_city VARCHAR(100),
    dest VARCHAR(10),
    dest_city VARCHAR(100),
    crs_dep_time INTEGER,
    dep_time INTEGER,
    dep_delay NUMERIC,
    crs_arr_time INTEGER,
    arr_time INTEGER,
    arr_delay NUMERIC,
    arr_del15 NUMERIC,
    cancelled NUMERIC,
    cancellation_code VARCHAR(5),
    crs_elapsed_time NUMERIC,
    distance NUMERIC
);

-- 2. Cleaned data, created by preprocess.py from "flights"
--    * Cancelled flights removed, day_of_week and month added, the two city
--    columns and the cancellation code dropped, blanks filled with 0.
--    526,882 rows. Used to train the delay model.
CREATE TABLE flights_processed (
    "FL_DATE" TIMESTAMP,
    "OP_UNIQUE_CARRIER" TEXT,
    "OP_CARRIER_FL_NUM" BIGINT,
    "ORIGIN" TEXT,
    "DEST" TEXT,
    "CRS_DEP_TIME" BIGINT,
    "DEP_TIME" DOUBLE PRECISION,
    "DEP_DELAY" DOUBLE PRECISION,
    "CRS_ARR_TIME" BIGINT,
    "ARR_TIME" DOUBLE PRECISION,
    "ARR_DELAY" DOUBLE PRECISION,
    "ARR_DEL15" DOUBLE PRECISION,       
    "CANCELLED" DOUBLE PRECISION,
    "CRS_ELAPSED_TIME" DOUBLE PRECISION,
    "DISTANCE" DOUBLE PRECISION,
    day_of_week INTEGER,                
    month INTEGER
);

-- 3. Model predictions on the test set, created by save_results.py
--    105,377 rows (the 20% of flights the model did not train on).
--    One row per test flight: the model inputs, the real result and the prediction.
--    The airline is stored as one boolean column per airline (one-hot encoding).
CREATE TABLE flight_predictions (
    day_of_week BIGINT,
    month BIGINT,
    "DISTANCE" DOUBLE PRECISION,
    "CRS_DEP_TIME" BIGINT,
    "OP_UNIQUE_CARRIER_9E" BOOLEAN,
    "OP_UNIQUE_CARRIER_AA" BOOLEAN,
    "OP_UNIQUE_CARRIER_AS" BOOLEAN,
    "OP_UNIQUE_CARRIER_B6" BOOLEAN,
    "OP_UNIQUE_CARRIER_DL" BOOLEAN,
    "OP_UNIQUE_CARRIER_F9" BOOLEAN,
    "OP_UNIQUE_CARRIER_G4" BOOLEAN,
    "OP_UNIQUE_CARRIER_HA" BOOLEAN,
    "OP_UNIQUE_CARRIER_MQ" BOOLEAN,
    "OP_UNIQUE_CARRIER_NK" BOOLEAN,
    "OP_UNIQUE_CARRIER_OH" BOOLEAN,
    "OP_UNIQUE_CARRIER_OO" BOOLEAN,
    "OP_UNIQUE_CARRIER_UA" BOOLEAN,
    "OP_UNIQUE_CARRIER_WN" BOOLEAN,
    "OP_UNIQUE_CARRIER_YX" BOOLEAN,
    actual_delayed DOUBLE PRECISION,    -- what really happened (1 = delayed)
    predicted_delayed DOUBLE PRECISION, -- what the model predicted (1 = delayed)
    prediction_label TEXT               -- 'On Time' or 'Delayed'
);

-- 4. Evaluation log, written by evaluate.py
--    One row per model per run. New runs are appended, old rows are kept.
CREATE TABLE model_results (
    run_time TIMESTAMP,
    model TEXT,
    accuracy DOUBLE PRECISION,
    "precision" DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    always_on_time_accuracy DOUBLE PRECISION  -- accuracy of always predicting "on time"
);