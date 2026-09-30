import os
import boto3


def download_model_from_s3():
    s3_bucket = os.getenv("S3_BUCKET_NAME", "my-ml-model-bucket")
    model_key = "churn/model.pt"

    print(f"Downloading model from S3 bucket: {s3_bucket}")
    s3_client = boto3.client("s3")
    # lệnh tải s3 về đĩa local
    # s3_client.download_file(s3_bucket, model_key, "model.pt")
    print(" Mock: Model download successfully from AWS S3!")


if __name__ == "__main__":
    download_model_from_s3()
