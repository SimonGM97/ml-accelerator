# terraform -chdir=terraform/ecr init: initialize Terraform & download the necessary provider plugins for AWS.
# terraform -chdir=terraform/ecr validate: validate Terraform configuration before applying it to ensure there are no syntax errors.
# terraform -chdir=terraform/ecr plan: shows what Terraform will do when applying the configuration (won’t make any changes).
# terraform -chdir=terraform/ecr apply: apply the configuration to create the resources.
# terraform -chdir=terraform/ecr destroy: delete all resources created by Terraform.

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
  description = "The AWS region where the ECR repository will be created."
  type        = string
}

variable "DOCKER_REPOSITORY_NAME" {
    description = "Name of the ECR repository that will be created."
    type        = string
}

# Define the provider (AWS)
provider "aws" {
  region = var.REGION
}

# Create an ECR repository
resource "aws_ecr_repository" "my_ecr_repo" {
  # ECR repository name
  name                     = var.DOCKER_REPOSITORY_NAME

  # If true, will delete the repository even if it contains images.
  force_delete             = true
  
  # The tag mutability setting for the repository
  image_tag_mutability = "MUTABLE"  # Optional: Set to IMMUTABLE for security best practice

  # Enable image scanning for vulnerabilities
  image_scanning_configuration {
    scan_on_push           = true
  }
  
  # Tags
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = var.ENV
  }
}