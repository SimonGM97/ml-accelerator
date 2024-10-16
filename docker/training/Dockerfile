# Use a base image compatible with arm64 architecture
# latest
FROM base_image_dev:v1.0.0

# Copy required script file into the container at /app
COPY scripts/training/training.py .

# Run the training.py script when the container launches
CMD [ "python", "training.py" ]