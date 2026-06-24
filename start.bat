@echo off
echo ===================================================
echo               DocMind AI RAG Chatbot
echo                Launcher for Windows
echo ===================================================
echo.

echo [1/3] Starting FastAPI Backend server...
start "DocMind Backend" cmd /k "backend\venv\Scripts\activate.bat && uvicorn backend.main:app --reload --port 8000"

echo [2/3] Starting React Frontend server...
start "DocMind Frontend" cmd /k "cd frontend && npm run dev"

echo [3/3] Opening browser in 5 seconds...
timeout /t 5 >nul
start http://localhost:5173

echo.
echo App is starting! 
echo - Backend: http://localhost:8000/docs (API Documentation)
echo - Frontend: http://localhost:5173 (Chat Platform)
echo.
echo To close the servers, simply close their command prompt windows.
echo ===================================================
pause
