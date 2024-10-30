#!/bin/bash
# chmod +x ./main.sh
# ./main.sh

# Unset environment variables
chmod +x ./scripts/bash/unset_env.sh
source ./scripts/bash/unset_env.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Update requirements.txt
# chmod +x ./scripts/bash/update_requirements.sh
# ./scripts/bash/update_requirements.sh

# Install yq (if necessary)
chmod +x ./scripts/bash/install_yq.sh
./scripts/bash/install_yq.sh

# Run unit & integrity Tests
# chmod +x ./scripts/bash/tests_running.sh
# ./scripts/bash/tests_running.sh

# Build docker repository
chmod +x ./scripts/bash/build_docker_repo.sh
./scripts/bash/build_docker_repo.sh

# Build Docker images
chmod +x ./scripts/bash/image_building.sh
./scripts/bash/image_building.sh

# Build infrastructure
chmod +x ./scripts/bash/build_infra.sh
./scripts/bash/build_infra.sh

# Run ETL workflow
chmod +x ./scripts/bash/etl.sh
./scripts/bash/etl.sh

# Run model building workflow
chmod +x ./scripts/bash/model_building.sh
./scripts/bash/model_building.sh

# Clean ports
# chmod +x ./scripts/bash/kill_ports.sh
# ./scripts/bash/kill_ports.sh

# Run apps
# chmod +x ./scripts/bash/app.sh
# ./scripts/bash/app.sh