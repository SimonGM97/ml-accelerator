# terraform -chdir=terraform/s3 init: initialize Terraform & download the necessary provider plugins for AWS.
# terraform -chdir=terraform/s3 validate: validate Terraform configuration before applying it to ensure there are no syntax errors.
# terraform -chdir=terraform/s3 plan: shows what Terraform will do when applying the configuration (won’t make any changes).
# terraform -chdir=terraform/s3 apply: apply the configuration to create the resources.
# terraform -chdir=terraform/s3 destroy: delete all resources created by Terraform.

# VARIABLES
variable "PROJECT_NAME" {
  description = "Name of the Project."
  type        = string
}

variable "VERSION" {
  description = "Version of the Project."
  type        = string
}

variable "ENV" {
    description = "Environment to create resources on."
    type        = string
}

variable "REGION_NAME" {
  description = "AWS region where the S3 bucket will be created."
  type        = string
}

variable "BUCKET_NAME" {
    description = "AWS Bucket name that will be created."
    type        = string
}

# AWS PROVIDER
provider "aws" {
  # Region name
  region = var.REGION_NAME
}

# S3 BUCKET
resource "aws_s3_bucket" "my_bucket" {
  # S3 bucket name
  bucket        = var.BUCKET_NAME

  # Indicates if all objects (including any locked objects) should be deleted from the bucket when the bucket is destroyed.
  force_destroy = false

  # Tags
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = var.ENV
  }
}
