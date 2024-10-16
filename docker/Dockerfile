# Define build arguments
ARG VERSION
ARG ENV

# Use a base image compatible with arm64 architecture
# latest
FROM ${ENV}-base-image:${VERSION} AS base

# Copy required files & directories into the container at /app
COPY .aws /root/.aws
COPY config config
COPY scripts/data_processing/data_processing.py .
COPY scripts/tuning/tuning.py .
COPY app.py .

# Run the app.py script when the container launches
# NOTE: The CMD can be re-written while running a container
CMD [ "python", "app.py" ]