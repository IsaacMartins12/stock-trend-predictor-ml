.PHONY: help up down build logs airflow-logs backfill reset

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================
# Infrastructure
# ============================================

build: ## Build Docker images
	docker compose build

up: ## Start all services (PostgreSQL + Airflow)
	docker compose up -d
	@echo ""
	@echo "============================================"
	@echo "  Services started:"
	@echo "  - PostgreSQL:      localhost:5432"
	@echo "  - Airflow UI:      http://localhost:8080"
	@echo "  - Airflow login:   admin / admin"
	@echo "============================================"

down: ## Stop all services
	docker compose down

reset: ## Reset everything (destroy volumes and rebuild)
	docker compose down -v
	docker compose build --no-cache
	docker compose up -d
	@echo "✓ Full reset complete"

logs: ## Show all container logs
	docker compose logs -f

airflow-logs: ## Show Airflow scheduler logs
	docker compose logs -f airflow-scheduler

db-logs: ## Show PostgreSQL logs
	docker compose logs -f postgres

# ============================================
# Pipeline Operations
# ============================================

backfill: ## Trigger historical backfill DAG (loads all data)
	docker compose exec airflow-webserver airflow dags trigger backfill_historical
	@echo "✓ Backfill DAG triggered. Check Airflow UI for progress."

trigger-bronze: ## Manually trigger bronze ingestion
	docker compose exec airflow-webserver airflow dags trigger bronze_ingestion

trigger-silver: ## Manually trigger silver transformation
	docker compose exec airflow-webserver airflow dags trigger silver_transformation

trigger-gold: ## Manually trigger gold feature engineering
	docker compose exec airflow-webserver airflow dags trigger gold_feature_engineering

# ============================================
# Development
# ============================================

install: ## Install Python dependencies locally
	pip install -r requirements.txt

dashboard: ## Run Streamlit dashboard locally
	streamlit run stream.py

api: ## Start FastAPI model serving
	docker compose up -d api
	@echo "✓ API running at http://localhost:8000"
	@echo "  Docs: http://localhost:8000/docs"

grafana: ## Start Grafana dashboard
	docker compose up -d grafana
	@echo "✓ Grafana running at http://localhost:3000"
	@echo "  Login: admin / admin"

db-shell: ## Open psql shell to data warehouse
	docker compose exec postgres psql -U stock_user -d stock_predictor

check-dags: ## Validate DAG files for syntax errors
	docker compose exec airflow-webserver airflow dags list
