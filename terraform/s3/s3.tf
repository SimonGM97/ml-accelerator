# terraform init: initialize Terraform & download the necessary provider plugins for AWS.
# terraform validate: validate Terraform configuration before applying it to ensure there are no syntax errors.
# terraform plan: shows what Terraform will do when applying the configuration (won’t make any changes).
# terraform apply: apply the configuration to create the resources.
# terraform destroy: delete all resources created by Terraform.

# Variables
variable "ENV" {
    description = "Environment to create resources on."
    type        = string
}

variable "REGION" {
  description = "The AWS region."
  type        = string
}

variable "BUCKET_NAME" {
    description = "AWS Bucket name that will be created."
    type        = string
}

# Define the provider (AWS)
provider "aws" {
  region = var.REGION
}

# Define the S3 bucket resource
resource "aws_s3_bucket" "my_bucket" {
  bucket = var.BUCKET_NAME

  # Tags
  tags = {
    Environment = var.ENV
  }
}
