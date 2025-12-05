#!/bin/bash

# ============================================================================
# Docker Setup Script for Trading ML System
# ============================================================================
# Usage: ./docker/start.sh [command]
# Commands: build, up, down, logs, clean, prod

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        print_warning "File .env not found. Creating from .env.example..."
        if [ ! -f "${PROJECT_ROOT}/.env.example" ]; then
            print_error ".env.example not found!"
            exit 1
        fi
        cp "${PROJECT_ROOT}/.env.example" "$ENV_FILE"
        print_warning "Please edit .env file and change passwords before running in production!"
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker not installed!"
        exit 1
    fi
    print_success "Docker is installed"

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not installed!"
        exit 1
    fi
    print_success "Docker Compose is installed"
}

# ============================================================================
# Commands
# ============================================================================

cmd_build() {
    print_header "Building Docker Images"
    check_env_file
    cd "$PROJECT_ROOT"
    docker-compose build "$@"
    print_success "Build completed"
}

cmd_up() {
    print_header "Starting Services"
    check_env_file
    cd "$PROJECT_ROOT"
    docker-compose up -d "$@"
    print_success "Services started"
    print_info "Waiting for services to be ready..."
    sleep 5
    cmd_status
}

cmd_down() {
    print_header "Stopping Services"
    cd "$PROJECT_ROOT"
    docker-compose down "$@"
    print_success "Services stopped"
}

cmd_logs() {
    print_header "Showing Logs"
    cd "$PROJECT_ROOT"
    if [ -z "$1" ]; then
        docker-compose logs -f --tail=100
    else
        docker-compose logs -f --tail=100 "$@"
    fi
}

cmd_status() {
    print_header "Service Status"
    cd "$PROJECT_ROOT"
    docker-compose ps

    echo -e "\n${BLUE}Service URLs:${NC}"
    echo -e "  Grafana:        ${YELLOW}http://localhost:3000${NC}"
    echo -e "  MLflow:         ${YELLOW}http://localhost:5000${NC}"
    echo -e "  Jupyter:        ${YELLOW}http://localhost:8888${NC}"
    echo -e "  Prometheus:     ${YELLOW}http://localhost:9090${NC}"
    echo -e "  Inference API:  ${YELLOW}http://localhost:8001${NC}"
    echo -e "  PostgreSQL:     ${YELLOW}localhost:5432${NC}"
}

cmd_logs_jupyter() {
    print_header "Getting Jupyter Token"
    cd "$PROJECT_ROOT"
    docker-compose logs jupyter | grep token || echo "Jupyter not running or token not found"
}

cmd_train() {
    print_header "Running Training Pipeline"
    check_env_file
    cd "$PROJECT_ROOT"
    docker-compose run --rm ml-training python -m src.pipelines.training_pipeline "$@"
}

cmd_test() {
    print_header "Running Tests"
    check_env_file
    cd "$PROJECT_ROOT"
    docker-compose run --rm ml-training pytest tests/ "$@"
}

cmd_clean() {
    print_header "Cleaning Up"
    print_warning "This will remove containers, volumes, and data!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        print_info "Cancelled"
        return
    fi

    cd "$PROJECT_ROOT"
    docker-compose down -v --remove-orphans
    print_success "Cleanup completed"
}

cmd_shell() {
    print_header "Opening Shell in ${1:-ml-training}"
    check_env_file
    cd "$PROJECT_ROOT"
    docker-compose exec "${1:-ml-training}" bash
}

cmd_postgres_backup() {
    print_header "Backing Up PostgreSQL Database"
    check_env_file
    cd "$PROJECT_ROOT"
    BACKUP_FILE="postgres_backup_$(date +%Y%m%d_%H%M%S).sql"
    docker-compose exec -T postgres pg_dump -U mlflow -d mlflow > "$BACKUP_FILE"
    print_success "Backup created: $BACKUP_FILE"
}

cmd_help() {
    cat << EOF
${BLUE}Trading ML System - Docker Management Script${NC}

${GREEN}Usage:${NC}
  ./docker/start.sh [command] [options]

${GREEN}Commands:${NC}
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

${GREEN}Examples:${NC}
  ./docker/start.sh build
  ./docker/start.sh up
  ./docker/start.sh logs ml-training
  ./docker/start.sh train --num-workers 4
  ./docker/start.sh test -vv
  ./docker/start.sh shell ml-training
  ./docker/start.sh status

${YELLOW}Notes:${NC}
  - Edit .env file to configure environment variables
  - Change passwords before production deployment
  - Use 'docker-compose' directly for advanced operations

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    local command="${1:-help}"

    check_docker

    case "$command" in
        build)
            cmd_build "${@:2}"
            ;;
        up)
            cmd_up "${@:2}"
            ;;
        down)
            cmd_down "${@:2}"
            ;;
        logs)
            cmd_logs "${@:2}"
            ;;
        status)
            cmd_status
            ;;
        train)
            cmd_train "${@:2}"
            ;;
        test)
            cmd_test "${@:2}"
            ;;
        shell)
            cmd_shell "${@:2}"
            ;;
        jupyter-token)
            cmd_logs_jupyter
            ;;
        db-backup)
            cmd_postgres_backup
            ;;
        clean)
            cmd_clean
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
