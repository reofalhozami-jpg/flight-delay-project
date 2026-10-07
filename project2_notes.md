# Project 2 Notes

## the goal
A chatbot that answers questions about the flight data, You type something
like "which airline has the most delays?" and it looks the answer up in the
Postgres database from Project 1 and replies.

## n8n
A tool where you build things by connecting boxes on a screen. Each box does
one job, like receiving the question, asking an AI model, or running a
database query. Not much code needed.

## Google ADK
Google's toolkit for building AI agents in Python. I write instructions for
the agent and give it a function that runs SQL on the database. It decides
when to call that function to answer a question.

## n8n vs ADK

I asked both chatbots the same questions and the answers matched. The one
difference was "which airport is the worst", where they used different
minimum flight counts for small airports, so each picked a different one.
I checked both in the database and both were right.

ADK got an airport name wrong, so I check names but trust the numbers.

n8n was faster to set up. ADK is more code, but gives me more control.