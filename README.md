# flight-delay-project
This project looks at US flight data from January 2024 to understand delays and cancellations

## Goal
Figure out which flights are likely to be delayed or cancelled and eventually build a model that can predict this

## Files
eda_flights.ipynb is the notebook with the data exploration covering delay rates cancellations and data quality issues


## Running the web app
1. Make sure the Postgres container is running (`docker ps` should show flight-postgres)
2. pip install flask (if not already installed)
3. python app.py
4. Open http://127.0.0.1:5000 in a browser

## Running the chatbot (ADK)
1. Start Docker Desktop and the flight-postgres container
2. Add your Gemini key to flight_agent/.env as GOOGLE_API_KEY=your key
3. Run `adk web` in the terminal
4. Open http://127.0.0.1:8000 and pick flight_agent from the dropdown
5. Ask a question about the flight data

