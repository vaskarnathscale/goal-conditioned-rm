"""AWS Utils."""
import os


def delete_s3_profile():
    if "AWS_PROFILE" in os.environ:
        del os.environ["AWS_PROFILE"]


def copy_save_directory_to_s3(s3, save_dir: str, s3_bucket: str, s3_path: str):
    """Copy save directory to s3."""
    for root, _, files in os.walk(save_dir):
        for file in files:
            local_path = os.path.join(root, file)
            formatted_s3_path = os.path.join(s3_path, os.path.relpath(local_path, save_dir))
            s3.upload_file(local_path, s3_bucket, formatted_s3_path)
