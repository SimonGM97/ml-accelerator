#!/bin/bash
# chmod +x ./scripts/bash/workflows/main.sh
# ./scripts/bash/workflows/main.sh

# Install yq (if necessary)
chmod +x ./scripts/bash/install_yq.sh
./scripts/bash/install_yq.sh

# Clean ports
chmod +x ./scripts/bash/kill_ports.sh
./scripts/bash/kill_ports.sh

# Run unit & integrity Tests
chmod +x ./scripts/bash/tests_running.sh
./scripts/bash/tests_running.sh

# Build Docker images
chmod +x ./scripts/bash/image_building.sh
./scripts/bash/image_building.sh

# Ruun model building workflow
chmod +x ./scripts/bash/model_building.sh
./scripts/bash/model_building.sh

# docker container run -it dev-base-image:v1.0.0