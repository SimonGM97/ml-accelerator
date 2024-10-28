# Define build arguments
ARG ENV=prod
ARG VERSION=latest

# Use a base image compatible with arm64 architecture
# latest
FROM ${ENV}-base-image:${VERSION} AS base


# Copy required files & directories into the container at /app
# Optionally, you can copy scripts into /opt/ml/code, instead of app/
COPY scripts scripts
COPY app.py .

# Define the entrypoint for the container
#   - Run the app.py script when the container launches
# ENTRYPOINT [ "python3", "app.py" ]

# NOTE: Both the ENTRYPOINT and the CMD can be re-written while running a container