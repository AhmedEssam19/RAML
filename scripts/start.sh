#!/bin/bash

uv run celery -A src worker --loglevel=INFO --concurrency=1 &

uv run fastapi run src/main.py
