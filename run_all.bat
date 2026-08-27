@echo off
echo Starting all Sanket services...

echo Starting Frontend (Next.js)...
start cmd /k "cd frontend && npm run dev"

echo Starting Avatar Backend (FastAPI)...
start cmd /k "cd Backend\hearing_to_deaf && .\venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000"

echo Starting Gesture Backend (Django)...
start cmd /k "cd Backend\deaf_to_hearing && .\run_app.bat"

echo All services have been launched in separate windows!
pause
