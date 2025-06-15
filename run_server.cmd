@echo off
CALL C:\ProgramData\anaconda3\Scripts\activate.bat d:\ronbun\.conda
python -m uvicorn main:app --reload --host 0.0.0.0 --port 5000
