@echo off
echo Starting Backend and Frontend...

:: Navigate to the backend directory and start the Flask server
cd backend
start cmd /k "python app.py"

:: Navigate to the frontend directory and start the Streamlit app
cd ../frontend
start cmd /k "python -m streamlit run app.py"

echo Both Backend and Frontend are running.
pause