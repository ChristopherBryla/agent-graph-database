# Genesys

Starter Python project scaffold.

This project now includes a minimal Docker-based pattern for running multiple
ML model services behind a single application gateway.

## Structure

- `src/genesys/`: application package and gateway API
- `services/ml_sample_model1/`: stub model container
- `tests/`: test suite
- `pyproject.toml`: project metadata and pytest config
- `docker-compose.yml`: local multi-container setup

## Docker architecture

- `app`: gateway service that lists models and proxies prediction requests
- `ml_sample_model1`: placeholder model service that exposes `/predict`

You can add more models later by copying the sample model service and setting a
matching `GENESYS_MODEL_<NAME>_URL` environment variable on the gateway.

## Getting started

1. Activate the virtual environment for this workspace.
2. Install the project and test dependencies:
   `python -m pip install -e .`
   `python -m pip install pytest`
3. Run tests:
   `python -m pytest`

## Run with Docker

1. Install Docker Desktop.
2. Start the stack:
   `docker compose up --build`
3. Open the gateway docs:
   `http://localhost:8000/docs`
4. Check the sample model directly:
   `http://localhost:8001/docs`

## Example request

Send a request through the gateway:

```bash
curl -X POST http://localhost:8000/predict/ml_sample_model1 \
  -H "Content-Type: application/json" \
  -d '{"payload": {"text": "sample input"}}'
```
