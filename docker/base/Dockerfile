# Use a python base image
FROM python:3.10.8

# Switch to /tmp dir
WORKDIR /tmp

# Upgrade pip
RUN pip install --upgrade pip

# Copy & install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install dask[dataframe]
RUN pip install "dask[dataframe]"

# Copy specified files to install PyTradeX
COPY ml_accelerator ml_accelerator
COPY README.md .
COPY LICENSE .
COPY setup.py .

# Install PyTradeX library
RUN pip install --no-cache-dir .

# Switch back to /app dir
WORKDIR /app

# Copy required files & directories into the container at /app
COPY .aws /root/.aws
COPY config config