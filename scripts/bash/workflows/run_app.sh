#!/bin/bash
# chmod +x ./scripts/bash/workflows/run_app.sh
# ./scripts/bash/workflows/run_app.sh

# Set environment variables
chmod +x ./scripts/bash/set_env_vars.sh
./scripts/bash/set_env_vars.sh

# Clean ports
# chmod +x ./scripts/bash/kill_ports.sh
# ./scripts/bash/kill_ports.sh

# Build Docker images
chmod +x ./scripts/bash/image_building.sh
./scripts/bash/image_building.sh

# Run docker-compose-app
chmod +x ./scripts/bash/app.sh
./scripts/bash/app.sh