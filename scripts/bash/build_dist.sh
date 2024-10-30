#!/bin/bash
# chmod +x ./scripts/bash/build_dist.sh
# ./scripts/bash/build_dist.sh

# Set environment variables
set -o allexport
source .env
set +o allexport

# Extract PROJECT_NAME & VERSION from config file
CONFIG_FILE="config/config.yaml"

PROJECT_NAME=$(yq eval '.PROJECT_PARAMS.PROJECT_NAME' ${CONFIG_FILE})
VERSION=$(yq eval '.PROJECT_PARAMS.VERSION' ${CONFIG_FILE})

# Set variables
ZIP_NAME="${PROJECT_NAME}-${VERSION}.zip"
DIST_DIR="dist"
INCLUDE_FILES=( 
    ".streamlit" 
    "config" 
    "docker" 
    "config"
    "docs"
    "ml_accelerator"
    "resources"
    "scripts"
    "terraform"
    "test"
    ".env.example"
    ".gitignore"
    "app.py"
    "LICENSE"
    "main.sh"
    "README.md"
    "requirements.in"
    "requirements.txt"
    "setup.py"
)

# Clean current dist directory
rm -rf $DIST_DIR/*

# Create the dist directory if it doesn't exist
mkdir -p "$DIST_DIR"

# Create the zip file with the specified files and directories
zip -r "$DIST_DIR/$ZIP_NAME" "${INCLUDE_FILES[@]}" \
    -x "*/__pycache__/*" "*/legal/*" "*.terraform/*" \
    "*terraform.lock.hcl*" "*terraform.tfstate*" "*terraform.tfstate.backup*" 
    # "*/.git/*"

# Check if the zip was successful
if [ $? -eq 0 ]; then
    echo "Package created successfully at $DIST_DIR/$ZIP_NAME"
else
    echo "Failed to create the package"
    exit 1
fi