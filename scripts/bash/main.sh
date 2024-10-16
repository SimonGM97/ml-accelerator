#!/bin/bash
# chmod +x ./scripts/bash/main.sh
# ./scripts/bash/main.sh

# Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo "yq command not found. yq will be installed using brew."

    brew install yq
    # sudo apt-get install yq
fi

# Run unit & integrity Tests
# chmod +x ./scripts/bash/tests_running.sh
# ./scripts/bash/tests_running.sh

# Run image_building.sh bash script
chmod +x ./scripts/bash/image_building.sh
./scripts/bash/image_building.sh

# Run model building workflow
chmod +x ./scripts/bash/model_building_workflow.sh
./scripts/bash/model_building_workflow.sh

# docker container run -it dev-base-image:v1.0.0