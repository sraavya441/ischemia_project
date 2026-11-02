@echo off
echo Starting CardioVision Backend and Dashboard...

echo Opening FastAPI Backend in a new window...
start cmd /k "python -m uvicorn api:app --reload"

echo Opening Streamlit Dashboard in a new window...
start cmd /k "python -m streamlit run dashboard.py"

echo Both servers are starting up! You can leave those windows running in the background.
