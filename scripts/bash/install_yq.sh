#!/bin/bash
# chmod +x ./scripts/bash/install_yq.sh
# ./scripts/bash/install_yq.sh

# Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo "yq command not found."

    # Detect the operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS. Installing yq using brew."

        # Check if brew is installed
        if ! command -v brew &> /dev/null
        then
            echo "Homebrew is not installed. Please install Homebrew first."
            exit 1
        else
            brew install yq
        fi

    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Detected Linux. Installing yq using apt-get."

        # Update package lists and install yq
        sudo apt-get update
        sudo apt-get install -y yq

    else
        echo "Unsupported OS: $OSTYPE"
        exit 1
    fi

# else
#     echo "yq is already installed."
fi