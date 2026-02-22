#!/bin/bash

echo "Running smoke tests..."

# Health check
status=$(curl -s http://localhost:8000/ | grep "ok")

if [[ "$status" == *"ok"* ]]; then
  echo "Health check passed ✅"
else
  echo "Health check failed ❌"
  exit 1
fi

echo "Smoke tests completed successfully!"