#!/bin/bash

echo "Pulling latest image..."
docker pull soumyaranmohanty/cats-dogs-api:latest

echo "Restarting service..."
docker-compose down
docker-compose up -d

echo "Deployment done!"