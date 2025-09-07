#!/bin/sh
# Wait for MinIO to start
sleep 5

# Configure alias for MinIO
mc alias set local http://minio:9000 minio minio123

# Create bucket if it does not exist
mc mb --ignore-existing local/mlflow

# Give public read access to the bucket
mc admin policy attach local readwrite --user minio

