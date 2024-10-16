# Use a base image compatible with arm64 architecture
# latest
FROM base-image:v1.0.0

# Copy required script file into the container at /app
COPY scripts/inference/inference.py .

# Run the inference.py script when the container launches
CMD [ "python", "inference.py" ]

# For production
# CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port","8000"]