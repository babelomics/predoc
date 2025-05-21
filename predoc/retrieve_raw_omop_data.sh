#!/bin/bash

source .env

# Check if required environment variables are set
if [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_RAW_OMOP_DATA" ] || [ -z "$LOCAL_DATA_DIR" ]; then
    echo "Error: Required personal environment variables are not set."
    echo "Please ensure REMOTE_USER, REMOTE_HOST, REMOTE_RAW_OMOP_DATA and LOCAL_DATA_DIR are set in the environment variables."
    exit 1
fi

# Create LOCAL_PATH directory if it doesn't exist
mkdir -p "$LOCAL_DATA_DIR/omop/raw/"

# Run rsync command
rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_RAW_OMOP_DATA/*" "$LOCAL_DATA_DIR/omop/raw/"

# Check rsync exit status
if [ $? -eq 0 ]; then
    echo "Data transfer completed successfully."
else
    echo "Error: Data transfer failed."
    exit 1
fi