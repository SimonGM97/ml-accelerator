# Define build arguments
ARG ENV=prod
ARG VERSION=latest
ARG BUCKET_NAME=ml-accelerator-bucket-prod

# Use a base image compatible with arm64 architecture
# latest
FROM ${ENV}-base-image:${VERSION} AS base

# Copy .env file
COPY .env .env

# Copy AWS credentials
COPY .aws /root/.aws

# Copy resources
COPY resources resources

# TRAINING JOBS directory convention:
# /opt/ml/input/: This is typically used for training jobs to store input data.
# /opt/ml/model/: For training jobs, this directory is where trained models are stored.
# /opt/ml/output/: This is used for outputs from processing or training jobs.

# Create training job directories inside the container
RUN mkdir -p /opt/ml/input && \
    mkdir -p /opt/ml/output && \
    mkdir -p /opt/ml/model

# PROCESSING JOBS directory convention:
# /opt/ml/processing/input/: This is where input data from S3 is downloaded.
# /opt/ml/processing/output/: This is where output data should be written (if your processing job produces results).
# Other optional outputs:
#   - /opt/ml/processing/train/: train datasets
#   - /opt/ml/processing/validation/: validation datasets
#   - /opt/ml/processing/test/: test datasets
#   - /opt/ml/processing/model/: model artifacts
#   - /opt/ml/processing/evaluation/: evaluation data

# Create processing job directories inside the container
RUN mkdir -p /opt/ml/processing/input && \
    mkdir -p /opt/ml/processing/output && \
    mkdir -p /opt/ml/processing/train && \
    mkdir -p /opt/ml/processing/validation && \
    mkdir -p /opt/ml/processing/test && \
    mkdir -p /opt/ml/processing/model && \
    mkdir -p /opt/ml/processing/evaluation

# Define volumes
VOLUME ["/app/${BUCKET_NAME}", "/app/config"]
VOLUME ["/opt/ml/input", "/opt/ml/output", "/opt/ml/model"]
VOLUME ["/opt/ml/processing/input", "/opt/ml/processing/output"]
VOLUME ["/opt/ml/processing/train", "/opt/ml/processing/validation", "/opt/ml/processing/test"]
VOLUME ["/opt/ml/processing/model", "/opt/ml/processing/evaluation"]

# Copy required files & directories into the container at /app
# Optionally, you can copy scripts into /opt/ml/code, instead of app/
COPY scripts scripts
COPY app.py .

# Run the app.py script when the container launches
# NOTE: The CMD can be re-written while running a container
CMD [ "python", "app.py" ]