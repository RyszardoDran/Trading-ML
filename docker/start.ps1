# ============================================================================
# Docker Setup Script for Trading ML System (PowerShell)
# ============================================================================
# Usage: .\docker\start.ps1 -Command "build|up|down|logs|status|clean"

param(
    [Parameter(Position = 0)]
    [ValidateSet("build", "up", "down", "logs", "status", "train", "test", "shell", "jupyter-token", "db-backup", "clean", "help")]
    [string]$Command = "help",
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

# ============================================================================
# Configuration
# ============================================================================
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$EnvFile = Join-Path $ProjectRoot ".env"
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"

# Colors
$Colors = @{
    Green  = 'Green'
    Red    = 'Red'
    Yellow = 'Yellow'
    Cyan   = 'Cyan'
}

# ============================================================================
# Functions
# ============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor $Colors.Cyan
    Write-Host $Message -ForegroundColor $Colors.Cyan
    Write-Host ("=" * 60) -ForegroundColor $Colors.Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $Colors.Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $Colors.Red
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $Colors.Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $Colors.Cyan
}

function Check-EnvFile {
    if (-not (Test-Path $EnvFile)) {
        Write-Warning-Custom "File .env not found. Creating from .env.example..."
        $ExampleEnvFile = "$ProjectRoot\.env.example"
        if (-not (Test-Path $ExampleEnvFile)) {
            Write-Error-Custom ".env.example not found!"
            exit 1
        }
        Copy-Item $ExampleEnvFile $EnvFile
        Write-Warning-Custom "Please edit .env file and change passwords before running in production!"
    }
}

function Check-Docker {
    try {
        docker --version | Out-Null
        Write-Success "Docker is installed"
    }
    catch {
        Write-Error-Custom "Docker not installed!"
        exit 1
    }

    try {
        docker-compose --version | Out-Null
        Write-Success "Docker Compose is installed"
    }
    catch {
        Write-Error-Custom "Docker Compose not installed!"
        exit 1
    }
}

# ============================================================================
# Commands
# ============================================================================

function Cmd-Build {
    Write-Header "Building Docker Images"
    Check-EnvFile
    Push-Location $ProjectRoot
    docker-compose build $Arguments
    Write-Success "Build completed"
    Pop-Location
}

function Cmd-Up {
    Write-Header "Starting Services"
    Check-EnvFile
    Push-Location $ProjectRoot
    docker-compose up -d $Arguments
    Write-Success "Services started"
    Write-Info "Waiting for services to be ready..."
    Start-Sleep -Seconds 5
    Cmd-Status
    Pop-Location
}

function Cmd-Down {
    Write-Header "Stopping Services"
    Push-Location $ProjectRoot
    docker-compose down $Arguments
    Write-Success "Services stopped"
    Pop-Location
}

function Cmd-Logs {
    Write-Header "Showing Logs"
    Push-Location $ProjectRoot
    if ($Arguments.Count -eq 0) {
        docker-compose logs -f --tail=100
    }
    else {
        docker-compose logs -f --tail=100 $Arguments
    }
    Pop-Location
}

function Cmd-Status {
    Write-Header "Service Status"
    Push-Location $ProjectRoot
    docker-compose ps
    Write-Host ""
    Write-Host "Service URLs:" -ForegroundColor $Colors.Cyan
    Write-Host "  Grafana:        http://localhost:3000" -ForegroundColor $Colors.Yellow
    Write-Host "  MLflow:         http://localhost:5000" -ForegroundColor $Colors.Yellow
    Write-Host "  Jupyter:        http://localhost:8888" -ForegroundColor $Colors.Yellow
    Write-Host "  Prometheus:     http://localhost:9090" -ForegroundColor $Colors.Yellow
    Write-Host "  Inference API:  http://localhost:8001" -ForegroundColor $Colors.Yellow
    Write-Host "  PostgreSQL:     localhost:5432" -ForegroundColor $Colors.Yellow
    Pop-Location
}

function Cmd-Logs-Jupyter {
    Write-Header "Getting Jupyter Token"
    Push-Location $ProjectRoot
    docker-compose logs jupyter | Select-String "token" -ErrorAction SilentlyContinue
    if ($null -eq $_) {
        Write-Info "Jupyter not running or token not found"
    }
    Pop-Location
}

function Cmd-Train {
    Write-Header "Running Training Pipeline"
    Check-EnvFile
    Push-Location $ProjectRoot
    docker-compose run --rm ml-training python -m src.pipelines.training_pipeline $Arguments
    Pop-Location
}

function Cmd-Test {
    Write-Header "Running Tests"
    Check-EnvFile
    Push-Location $ProjectRoot
    docker-compose run --rm ml-training pytest tests/ $Arguments
    Pop-Location
}

function Cmd-Shell {
    Write-Header "Opening Shell"
    Check-EnvFile
    Push-Location $ProjectRoot
    $ServiceName = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "ml-training" }
    docker-compose exec $ServiceName bash
    Pop-Location
}

function Cmd-Clean {
    Write-Header "Cleaning Up"
    Write-Warning-Custom "This will remove containers, volumes, and data!"
    $response = Read-Host "Are you sure? (yes/no)"
    if ($response -ne "yes") {
        Write-Info "Cancelled"
        return
    }

    Push-Location $ProjectRoot
    docker-compose down -v --remove-orphans
    Write-Success "Cleanup completed"
    Pop-Location
}

function Cmd-Db-Backup {
    Write-Header "Backing Up PostgreSQL Database"
    Check-EnvFile
    Push-Location $ProjectRoot
    $BackupFile = "postgres_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql"
    docker-compose exec -T postgres pg_dump -U mlflow -d mlflow | Out-File $BackupFile
    Write-Success "Backup created: $BackupFile"
    Pop-Location
}

function Cmd-Help {
    $HelpText = @"
Trading ML System - Docker Management Script

Usage:
  .\docker\start.ps1 -Command [command] [options]

Commands:
  build              Build Docker images
  up                 Start all services in background
  down               Stop all services
  status             Show service status
  logs [service]     Follow logs (default: all)
  train [options]    Run training pipeline
  test [options]     Run pytest tests
  shell [service]    Open bash shell in container (default: ml-training)
  jupyter-token      Get Jupyter access token
  db-backup          Backup PostgreSQL database
  clean              Remove all containers and volumes (DESTRUCTIVE)
  help               Show this help message

Examples:
  .\docker\start.ps1 -Command build
  .\docker\start.ps1 -Command up
  .\docker\start.ps1 -Command logs -Arguments ml-training
  .\docker\start.ps1 -Command train
  .\docker\start.ps1 -Command test
  .\docker\start.ps1 -Command status

Notes:
  - Edit .env file to configure environment variables
  - Change passwords before production deployment
  - Use 'docker-compose' directly for advanced operations
"@
    Write-Host $HelpText
}

# ============================================================================
# Main
# ============================================================================

Check-Docker

switch ($Command) {
    "build" { Cmd-Build }
    "up" { Cmd-Up }
    "down" { Cmd-Down }
    "logs" { Cmd-Logs }
    "status" { Cmd-Status }
    "train" { Cmd-Train }
    "test" { Cmd-Test }
    "shell" { Cmd-Shell }
    "jupyter-token" { Cmd-Logs-Jupyter }
    "db-backup" { Cmd-Db-Backup }
    "clean" { Cmd-Clean }
    "help" { Cmd-Help }
    default {
        Write-Error-Custom "Unknown command: $Command"
        Cmd-Help
        exit 1
    }
}
