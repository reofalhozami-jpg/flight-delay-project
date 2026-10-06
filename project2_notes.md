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
