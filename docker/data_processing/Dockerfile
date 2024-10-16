# Define build arguments
ARG VERSION

# Use a base image compatible with arm64 architecture
# latest
FROM base-image:${VERSION} AS base

# Copy required script file into the container at /app
COPY scripts/data_processing/data_processing.py .

# Run the data_processing.py script when the container launches
CMD [ "python", "data_processing.py" ]