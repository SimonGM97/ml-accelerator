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
  
  # ECR repository tags
  tags = {
    Name                   = var.DOCKER_REPOSITORY_NAME
    Environment            = var.ENV
  }
}