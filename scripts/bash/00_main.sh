#!/bin/bash
# chmod +x ./scripts/bash/00_main.sh
# ./scripts/bash/00_main.sh

# Install yq (if necessary)
chmod +x ./scripts/bash/01_install_yq.sh
./scripts/bash/01_install_yq.sh

# Run unit & integrity Tests
chmod +x ./scripts/bash/02_tests_running.sh
./scripts/bash/02_tests_running.sh

# Build Docker images
chmod +x ./scripts/bash/03_image_building.sh
./scripts/bash/03_image_building.sh

# Ruun model building workflow
chmod +x ./scripts/bash/04_model_building_workflow.sh
./scripts/bash/04_model_building_workflow.sh

# docker container run -it dev-base-image:v1.0.0