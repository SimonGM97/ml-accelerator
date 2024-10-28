#!/bin/bash
# chmod +x ./scripts/bash/update_requirements.sh
# ./scripts/bash/update_requirements.sh

# Install pip-tools
pip install pip-tools

# Compile and update requirements.txt
pip-compile --upgrade

# Re-install library
pip install -e .