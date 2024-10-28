#!/bin/bash
# chmod +x ./main.sh
# ./main.sh build_new_images | dont_build_new_image

# Deactivate conda
# conda deactivate

# Set environment variables
set -o allexport
source .env
set +o allexport

# Install yq (if necessary)
chmod +x ./scripts/bash/install_yq.sh
./scripts/bash/install_yq.sh

# Update requirements.txt
# chmod +x ./scripts/bash/update_requirements.sh
# ./scripts/bash/update_requirements.sh

# Clean ports
# chmod +x ./scripts/bash/kill_ports.sh
# ./scripts/bash/kill_ports.sh

# Run unit & integrity Tests
# chmod +x ./scripts/bash/tests_running.sh
# ./scripts/bash/tests_running.sh

# Build Docker images
chmod +x ./scripts/bash/image_building.sh
./scripts/bash/image_building.sh $1

# Build infrastructure
chmod +x ./scripts/bash/build_infra.sh
./scripts/bash/build_infra.sh

# Run ETL workflow
chmod +x ./scripts/bash/etl.sh
./scripts/bash/etl.sh

# Run model building workflow
# chmod +x ./scripts/bash/model_building.sh
# ./scripts/bash/model_building.sh

# Run apps
# chmod +x ./scripts/bash/app.sh
# ./scripts/bash/app.sh

# docker container run -it dev-base-image:v1.0.0