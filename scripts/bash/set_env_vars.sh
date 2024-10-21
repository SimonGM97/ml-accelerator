#!/bin/bash
# chmod +x ./scripts/bash/set_env_vars.sh
# ./scripts/bash/set_env_vars.sh

# Set environment variables
set -o allexport
source .env
set +o allexport