# PowerShell script for running PointCloudSeg pipeline

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Starting PointCloudSeg Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Write-Host "[Step 1/4] Installing requirements..." -ForegroundColor Yellow
# pip install -r requirements.txt
# if ($LASTEXITCODE -ne 0) {
#     Write-Host "Error installing requirements" -ForegroundColor Red
#     exit 1
# }

Write-Host ""
Write-Host "[Step 2/4] Running data preprocessor..." -ForegroundColor Yellow
python -m preprocessor.datapreprocessor
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error running preprocessor" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[Step 3/4] Running EDA analysis..." -ForegroundColor Yellow
python -m eda.eda
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error running EDA" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[Step 4/4] Starting training (will run for 3 seconds)..." -ForegroundColor Yellow
$job = Start-Job -ScriptBlock { python -m train.trainer }
Wait-Job $job -Timeout 3 | Out-Null
Stop-Job $job
Remove-Job $job

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Pipeline execution completed!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
