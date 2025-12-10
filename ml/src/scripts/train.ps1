<#
.SYNOPSIS
  Convenience PowerShell script to run the XAU/USD training pipeline or a data health check.

.DESCRIPTION
  Wraps Python execution with proper PYTHONPATH setup so tests and imports resolve.
  Supports optional parameters for target construction and health-check mode.

.PARAMETER Horizon
  Forward window in minutes for target creation (default 5).

.PARAMETER MinReturnBp
  Minimum cumulative return threshold in basis points (default 5.0).

.PARAMETER HealthCheckDir
  If provided, runs the health check on the given directory and exits.

.EXAMPLE
  ./train.ps1 -Horizon 10 -MinReturnBp 8

.EXAMPLE
  ./train.ps1 -HealthCheckDir "c:\\path\\to\\data"
#>
[CmdletBinding()] param(
  [int]$Horizon = 5,
  [double]$MinReturnBp = 5.0,
  [string]$HealthCheckDir,
  [switch]$SetupEnv
)

$ErrorActionPreference = 'Stop'

# Resolve repository paths
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$MlRoot = Join-Path $RepoRoot 'ml'
$PipelinePath = Join-Path $MlRoot 'src\pipelines\training_pipeline.py'
$RequirementsPath = Join-Path $MlRoot 'requirements_ml.txt'

# Ensure Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Error 'Python not found on PATH. Please install Python 3.10+ and ensure it is available.'
}

# Set PYTHONPATH so `from src...` imports resolve
$env:PYTHONPATH = $MlRoot

# Optional environment setup: install ML requirements if missing
if ($SetupEnv) {
  Write-Host "Setting up Python environment (requirements)" -ForegroundColor Cyan
  if (Test-Path $RequirementsPath) {
    pip --version > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
      Write-Warning 'pip not found. Attempting to use python -m pip.'
      python -m pip --version | Out-Null
      if ($LASTEXITCODE -ne 0) {
        Write-Error 'pip is not available. Please install pip.'
        exit 1
      }
      python -m pip install -r $RequirementsPath
    }
    else {
      pip install -r $RequirementsPath
    }
  }
  else {
    Write-Warning "Requirements file not found at: $RequirementsPath"
  }
}

# Proactive check for critical dependency (xgboost); install if missing
python -c "import xgboost" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Warning 'Python package xgboost not found. Attempting installation from requirements.'
  if (Test-Path $RequirementsPath) {
    python -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) {
      Write-Error 'Failed to install required packages. Please install dependencies in ml/requirements_ml.txt.'
      exit 1
    }
  }
  else {
    python -m pip install xgboost
    if ($LASTEXITCODE -ne 0) {
      Write-Error 'Failed to install xgboost. Please ensure internet connectivity or install manually.'
      exit 1
    }
  }
}

if ($HealthCheckDir) {
  Write-Host "Running health check for: $HealthCheckDir" -ForegroundColor Cyan
  python $PipelinePath --health-check-dir $HealthCheckDir
  exit $LASTEXITCODE
}

Write-Host "Running training pipeline (Horizon=$Horizon, MinReturnBp=$MinReturnBp bp)" -ForegroundColor Cyan
python $PipelinePath --horizon $Horizon --min-return-bp $MinReturnBp

exit $LASTEXITCODE
