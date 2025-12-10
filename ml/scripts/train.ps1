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
  [string]$HealthCheckDir
)

$ErrorActionPreference = 'Stop'

# Resolve repository paths
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$MlRoot = Join-Path $RepoRoot 'ml'
$PipelinePath = Join-Path $MlRoot 'src\pipelines\training_pipeline.py'

# Ensure Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Error 'Python not found on PATH. Please install Python 3.10+ and ensure it is available.'
}

# Set PYTHONPATH so `from src...` imports resolve
$env:PYTHONPATH = $MlRoot

if ($HealthCheckDir) {
  Write-Host "Running health check for: $HealthCheckDir" -ForegroundColor Cyan
  python $PipelinePath --health-check-dir $HealthCheckDir
  exit $LASTEXITCODE
}

Write-Host "Running training pipeline (Horizon=$Horizon, MinReturnBp=$MinReturnBp bp)" -ForegroundColor Cyan
python $PipelinePath --horizon $Horizon --min-return-bp $MinReturnBp

exit $LASTEXITCODE
