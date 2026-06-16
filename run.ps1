$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "backend\requirements.txt"

if (-not (Test-Path $Python)) {
    Write-Host "Creating project virtual environment..." -ForegroundColor Cyan
    py -3.14 -m venv (Join-Path $ProjectRoot ".venv")
}

& $Python -c "import fastapi, uvicorn, websockets, cv2, mediapipe, sklearn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required packages..." -ForegroundColor Cyan
    & $Python -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
}

Write-Host "Starting Sign Language Detector..." -ForegroundColor Green
Write-Host "Open http://localhost:5000 in your browser." -ForegroundColor Yellow
Set-Location (Join-Path $ProjectRoot "backend")
& $Python -m uvicorn app:app --host 127.0.0.1 --port 5000
