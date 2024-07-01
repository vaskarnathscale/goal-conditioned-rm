""""Governs moving models from Scale's S3."""
import os
from urllib.parse import urlparse

import boto3


def check_if_s3_and_copy(path: str, client: boto3.client):
    """Check if S3 and copy everything at directory to a local path."""
    if path.startswith("s3://"):
        # Parse the S3 URL to bucket and prefix
        print(f"Copying {path} to local directory")
        parsed_url = urlparse(path)
        bucket_name = parsed_url.netloc
        prefix = parsed_url.path.lstrip('/')
        
        # Define a local directory to save the files
        local_dir = os.path.join("/tmp", bucket_name, prefix)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        
        # List and copy each file from the S3 directory to the local directory
        paginator = client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    file_name = file_key.split('/')[-1]  # Extract the file name
                    local_file_path = os.path.join(local_dir, file_name)
                    
                    # Download the file
                    print(f"Downloading {file_key} to {local_file_path}")
                    client.download_file(bucket_name, file_key, local_file_path)
        
        # Return the local directory path where files were copied
        return local_dir

    # If the path does not start with "s3://", return it as is
    return path
