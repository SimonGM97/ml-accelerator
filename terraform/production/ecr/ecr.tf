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
  description = "The AWS region where the ECR repository will be created."
  type        = string
}

variable "DOCKER_REPOSITORY_NAME" {
  description = "Name of the ECR repository that will be created (on Prod environment)."
  type        = string
}

# Define the provider (AWS)
provider "aws" {
  region = var.REGION_NAME
}

# ECR REPOSITORY
resource "aws_ecr_repository" "ecr_repository_prod" {
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
    Project       = var.PROJECT_NAME
    Version       = var.VERSION
    Environment = "prod"
  }
}