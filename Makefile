# AI Training Application Makefile

.PHONY: help build clean dev prod logs stop restart

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build production containers
	docker-compose build

dev: ## Start development environment with hot reload
	docker-compose -f docker-compose.dev.yml up --build

prod: ## Start production environment
	docker-compose up --build -d

logs: ## View logs from all services
	docker-compose logs -f

backend-logs: ## View backend logs only
	docker-compose logs -f backend

frontend-logs: ## View frontend logs only
	docker-compose logs -f frontend

stop: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

clean: ## Remove all containers, volumes, and images
	docker-compose down -v --rmi all

backend-shell: ## Open shell in backend container
	docker-compose exec backend bash

frontend-shell: ## Open shell in frontend container (prod)
	docker-compose exec frontend sh

install-local: ## Install dependencies locally (without Docker)
	cd frontend/traninging && npm install
	cd ../..
	cd backend && pip install -r requirements.txt

run-local: ## Run both services locally (requires local setup)
	@echo "Starting backend..."
	@cd backend && python main.py &
	@echo "Starting frontend..."
	@cd frontend/traninging && npm start &
	@echo "Both services started. Press Ctrl+C to stop."
	@trap "echo 'Stopping services...'; pkill -f 'python main.py'; pkill -f 'ng serve'; exit" INT TERM; wait
