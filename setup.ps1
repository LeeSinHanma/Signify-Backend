param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating/using virtual environment..."
if (-not (Test-Path -LiteralPath ".\venv")) {
    & $PythonExe -m venv venv
}

$activateScript = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path -LiteralPath $activateScript)) {
    throw "Could not find venv activation script at $activateScript"
}

Write-Host "Activating virtual environment..."
. $activateScript

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing project dependencies..."
python -m pip install -r requirements.txt

Write-Host "\nSetup complete. Next steps:"
Write-Host "1. Run: python train.py"
Write-Host "2. Run backend: python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000"
Write-Host "3. Open: http://127.0.0.1:8000/docs"
