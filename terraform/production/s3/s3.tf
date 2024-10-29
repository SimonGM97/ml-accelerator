# VARIABLES
variable "PROJECT_NAME" {
  description = "Name of the Project."
  type        = string
}

variable "VERSION" {
  description = "Version of the Project."
  type        = string
}

variable "REGION_NAME" {
  description = "AWS region where the S3 bucket will be created."
  type        = string
}

variable "BUCKET_NAME" {
    description = "AWS Bucket name that will be created (for Prod environment)."
    type        = string
}

# AWS PROVIDER
provider "aws" {
  # Region name
  region = var.REGION_NAME
}

# S3 PROD BUCKET
resource "aws_s3_bucket" "s3_bucket_prod" {
  # S3 bucket name
  bucket        = var.BUCKET_NAME

  # Indicates if all objects (including any locked objects) should be deleted from the bucket when the bucket is destroyed.
  #   - Set to false for security reasons
  force_destroy = false

  # Tags
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = "prod"
  }
}

