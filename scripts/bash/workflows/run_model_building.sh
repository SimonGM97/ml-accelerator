#!/bin/bash
# chmod +x ./scripts/bash/workflows/run_model_building.sh
# ./scripts/bash/workflows/run_model_building.sh

# Build Docker images
chmod +x ./scripts/bash/image_building.sh
./scripts/bash/image_building.sh

# Run docker-compose-model-building
chmod +x ./scripts/bash/model_building.sh
./scripts/bash/model_building.sh