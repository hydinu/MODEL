# =============================================================================
# Makefile — Developer shortcuts for the Crowd Detection production stack
# =============================================================================

.PHONY: build up down restart logs ps clean shell-api shell-db seed test

COMPOSE = docker compose
ENV_FILE = .env

## Build all service images
build:
	$(COMPOSE) --env-file $(ENV_FILE) build

## Start all services (detached)
up:
	$(COMPOSE) --env-file $(ENV_FILE) up -d

## Stop all services
down:
	$(COMPOSE) --env-file $(ENV_FILE) down

## Stop + remove volumes (DESTRUCTIVE — wipes MongoDB data)
down-v:
	$(COMPOSE) --env-file $(ENV_FILE) down -v

## Restart a specific service: make restart svc=api
restart:
	$(COMPOSE) --env-file $(ENV_FILE) restart $(svc)

## Tail logs for all services
logs:
	$(COMPOSE) --env-file $(ENV_FILE) logs -f

## Tail logs for a specific service: make logs-svc svc=api
logs-svc:
	$(COMPOSE) --env-file $(ENV_FILE) logs -f $(svc)

## Show running containers
ps:
	$(COMPOSE) --env-file $(ENV_FILE) ps

## Open a shell inside the API container
shell-api:
	$(COMPOSE) --env-file $(ENV_FILE) exec api /bin/sh

## Open mongosh inside the MongoDB container
shell-db:
	$(COMPOSE) --env-file $(ENV_FILE) exec mongodb mongosh \
		-u $$(grep MONGO_ROOT_USER $(ENV_FILE) | cut -d= -f2) \
		-p $$(grep MONGO_ROOT_PASS $(ENV_FILE) | cut -d= -f2) \
		--authenticationDatabase admin

## Remove all stopped containers + dangling images
clean:
	docker system prune -f

## Quick health check against the running API
health:
	curl -s http://localhost/health | python -m json.tool

## Run tests inside the API container
test:
	$(COMPOSE) --env-file $(ENV_FILE) exec api python -m pytest tests/ -v
