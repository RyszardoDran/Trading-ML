# Docker Setup for XAU/USD Trading ML System

Prosta konfiguracja Docker do pracy z ML pipeline'em.

## Usługi

| Usługa | Port | Opis |
|--------|------|------|
| **ml-training** | - | Training pipeline |
| **jupyter** | 8888 | Development environment (EDA, notebooks) |

## Setup

### Nowe zmienne \.env

Dodane zostały dwie obowiązkowe zmienne środowiskowe wykorzystywane przez pipeline:

| Nazwa | Domyślna wartość | Opis |
|-------|------------------|------|
| `PYTHONPATH` | `/app` | Gwarantuje widoczność modułu `src` oraz poprawne działanie `python -m src....` |
| `DATA_DIR` | `/app/src/data` | Ścieżka do surowych danych XAU/USD montowanych w kontenerze |

Plik `.env.example` został zaktualizowany, więc wystarczy ponownie skopiować go do `.env` aby mieć prawidłowe ustawienia.

### 1. Przygotowanie

```bash
# Skopiuj .env
cp .env.example .env

# Edytuj jeśli potrzeba
nano .env  # lub notepad .env na Windows
```

### 2. Budowanie

```bash
# Build image
docker-compose build

# Lub konkretny serwis
docker-compose build ml-training
docker-compose build jupyter
```

### 3. Uruchomienie

```bash
# Start all services
docker-compose up -d

# Lub tylko Jupyter do pracy
docker-compose up -d jupyter

# Lub tylko training
docker-compose up -d ml-training

# Status
docker-compose ps

# Logi
docker-compose logs -f ml-training
docker-compose logs -f jupyter
```

### 4. Dostęp

```
📓 Jupyter Lab: http://localhost:8888
```

## Operacje

### Training

```bash
# Uruchomienie training pipeline
docker-compose run ml-training python -m src.pipelines.training_pipeline

# Lub z argumentami
docker-compose run ml-training python -m src.pipelines.training_pipeline --epochs 50
```

### Jupyter

```bash
# Zdobycie tokenu
docker-compose logs jupyter | grep token

# Otwórz w przeglądarce
# http://localhost:8888?token=YOUR_TOKEN
```

### Shell w kontenerze

```bash
docker-compose exec ml-training bash
docker-compose exec jupyter bash
```

### Testy

```bash
docker-compose run ml-training pytest tests/
docker-compose run ml-training pytest tests/test_training_pipeline.py
docker-compose run ml-training pytest tests/ -vv
```

## Czyszczenie

```bash
# Stop services
docker-compose stop

# Remove containers
docker-compose rm

# Remove everything (with volumes)
docker-compose down -v
```

## Troubleshooting

### Port 8888 zajęty?

```bash
# Zmień port w docker-compose.yml
# Lub zabij proces
lsof -i :8888
kill -9 <PID>
```

### Import bibliotek nie działa?

```bash
# Sprawdź czy zainstalowały się
docker-compose exec ml-training pip list

# Reinstall requirements
docker-compose exec ml-training pip install -r requirements_ml.txt

# Rebuild image
docker-compose build --no-cache ml-training
```

### Out of Memory?

Zwiększ limit w docker-compose.yml:
```yaml
services:
  ml-training:
    deploy:
      resources:
        limits:
          memory: 8G
```

## Workflow Development

1. **Edytuj kod lokalnie** (`./ml/src/`) - zmienia się automatycznie w kontenerze
2. **Otwórz Jupyter** - `http://localhost:8888`
3. **Eksperymentuj** - wszystkie zmiany są persystentne
4. **Run training** - `docker-compose run ml-training ...`
5. **Check results** - modele w `./ml/models/`

---

**Potrzebujesz więcej?** Dodamy Prometheus, Grafana, MLflow później.

