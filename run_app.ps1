Set-Location $PSScriptRoot
if (Test-Path ".\.venv\Scripts\streamlit.exe") {
    .\.venv\Scripts\streamlit.exe run app/ui.py
} else {
    Write-Host "Streamlit environment not found. Please create the virtual environment first."
    Read-Host "Press Enter to exit"
}
