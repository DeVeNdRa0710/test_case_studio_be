#!/bin/bash
set -e

# Create venv only if not exists
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

# Run migrations (only if configured)
if [ -f "alembic.ini" ]; then
  alembic upgrade head
else
  echo "Skipping migrations (no alembic.ini found)"
fi

# Kill existing process on port 8080
fuser -k 8080/tcp || true

# Start server
uvicorn app.main:app --reload --port 8080