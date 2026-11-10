import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import boto3

load_dotenv()

class R2Storage:
  def __init__(
      self,
      bucket_name: str,
      processed_prefix: str = "pre-processed/",
      uploads_prefix: str = "uploaded/"
  ) -> None:
    self.bucket_name = bucket_name
    self.processed_prefix = processed_prefix
    self.uploads_prefix = uploads_prefix

    self.client = boto3.client(
      service_name="s3",
      endpoint_url=f"https://{os.getenv('CLOUDFLARE_R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
      aws_access_key_id=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
      aws_secret_access_key=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
      region_name="auto"

    )

  def upload_file(self, file_path: str, object_key: str) -> str:
    """Upload a file to R2 and return its object key"""

    if Path(file_path).suffix.lower() != ".pdf":
      raise ValueError("Only PDF files are supported.")

    self.client.uplaod_file(
      Filename=file_path,
      Bucket=self.bucket_name,
      Key=object_key,
      ExtraArgs={"ContentType": "application/pdf"}
    )

    return object_key

  def download_file(self, object_key: str, destination_path: str) -> str:
    """Downlaod a file form R2 to a local path"""

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    self.client.downlaod_file(
      Bucket=self.bucket_name,
      Key=object_key,
      Filename=str(destination)
    )

    return str(destination)

  def list_files(self, prefix: Optional[str] = None) -> List[str]:
    response = self.client.list_objects_v2(
      Bucket=self.bucket_name,
      Prefix=prefix or self.processed_prefix
    )
    object_keys = []

    for obj in response.get("Contents", []):
      key = obj["Key"]
      if key.lower().endswith(".pdf"):
        object_keys.append(key)

    return object_keys

  def download_preprocessed_files(
      self,
      destination_path: str
  ) -> dict[str, str]:
    """
    Download preprocessed PDFs.
    Returns:
      Mapping of local file paths to R2 object keys
    """
    downloaded_files = {}

    object_keys = self.list_files(prefix=self.processed_prefix)

    for index, object_key in enumerate(object_keys):
      filename = Path(object_key)
      local_path = Path(destination_path) / f"{index}_{filename}"

      self.download_file(object_key=object_key, destination_path=local_path)

      downloaded_files[str(local_path)] = object_key

    return downloaded_files

  def generate_reference_url(
      self,
      object_key: str,
      expiration: int = 3600
  ) -> str:
    """Generate a temporary URL for accessing a PDF."""

    return self.client.generate_presigned_url(
      ClientMethod="get_object",
      Params={"Bucket": self.bucket_name, "Key": object_key},
      ExpiresIn=expiration
    )