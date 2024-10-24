# terraform -chdir=terraform/iam init: initialize Terraform & download the necessary provider plugins for AWS.
# terraform -chdir=terraform/iam validate: validate Terraform configuration before applying it to ensure there are no syntax errors.
# terraform -chdir=terraform/iam plan: shows what Terraform will do when applying the configuration (won’t make any changes).
# terraform -chdir=terraform/iam apply: apply the configuration to create the resources.
# terraform -chdir=terraform/iam destroy: delete all resources created by Terraform.

# Variables
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

variable "REGION" {
  description = "The AWS region where the S3 bucket will be created."
  type        = string
}

variable "SAGEMAKER_EXECUTION_ROLE_NAME" {
    description = "Execution role name to run SageMaker pipelines."
    type        = string
}

# Define the AWS provider
provider "aws" {
  region = var.REGION
}

# Define the IAM role for SageMaker
resource "aws_iam_role" "sagemaker_execution_role" {
  name = var.SAGEMAKER_EXECUTION_ROLE_NAME
  
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = var.ENV
  }

  assume_role_policy = jsonencode({
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
          "Service": "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# Attach managed policy for basic SageMaker permissions
resource "aws_iam_role_policy_attachment" "sagemaker_basic_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Attach managed policy for S3 access (modify this based on your needs)
resource "aws_iam_role_policy_attachment" "s3_access_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# Attach managed policy for CloudWatch logs (optional, for logging purposes)
resource "aws_iam_role_policy_attachment" "cloudwatch_access_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}
