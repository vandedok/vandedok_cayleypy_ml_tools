import requests
import subprocess
import os
from pathlib import Path




def get_r2_bucket_usage_with_api(account_id: str, bucket_name: str, api_token: str) -> dict:
    """
    Get R2 bucket usage using Cloudflare API.
    Note: api size request has some delay, so the size returned may be not up-to-date.
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/buckets/{bucket_name}/usage"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)

    json_data = response.json()
    result = json_data['result']

    # payload_size = result['payloadSize']
    # metadata_size = result['metadataSize']
    # object_count = result['objectCount']

    return result


def sync_s3_bucket(
        # cf_account_id: str,
        s3_provider: str,
        s3_endpoint: str,
        s3_access_key: str,
        s3_secret_access_key: str,
        path_to_dir: Path, 
        # path_rclone_config: str, 
        bucket_name: str,
        path_in_bucket: str="",
        timeout: int=300,
        exclude_patterns: list[str]=[],
        ) -> None:
        
    # TODO: add timeout
    base_env = os.environ.copy()
    path_to_dir = Path(path_to_dir)
    # path_rclone_config = Path(path_rclone_config)

    if not path_in_bucket:
        path_in_bucket = path_to_dir.name
    else:
        path_in_bucket = str(path_in_bucket)

    # subprocess.run(["rclone", "--config", path_rclone_config, "tree", f"r2_cayleypy:{bucket_name}"], env=base_env)

    command = [
        "rclone",
        "--s3-provider", s3_provider,
        "--s3-endpoint", s3_endpoint,
        "--s3-access-key-id", s3_access_key,
        "--s3-secret-access-key", s3_secret_access_key,
        "sync", str(path_to_dir), f":s3:{bucket_name}/{path_in_bucket}",
    ]

    for pattern in exclude_patterns:
        command = command + ["--exclude", pattern]

    print("Running command:", " ".join(command))
    subprocess.run(command, env=base_env, timeout=timeout)