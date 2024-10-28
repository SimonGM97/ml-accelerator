# terraform -chdir=terraform/step_functions init: initialize Terraform & download the necessary provider plugins for AWS.
# terraform -chdir=terraform/step_functions validate: validate Terraform configuration before applying it to ensure there are no syntax errors.
# terraform -chdir=terraform/step_functions plan: shows what Terraform will do when applying the configuration (won’t make any changes).
# terraform -chdir=terraform/step_functions apply: apply the configuration to create the resources.
# terraform -chdir=terraform/step_functions destroy: delete all resources created by Terraform.

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
  description = "AWS region where the resources will be created."
  type        = string
}

variable "BUCKET_NAME" {
  description = "S3 bucket name where processing datasets will be stored."
  type        = string
}

variable "SAGEMAKER_EXECUTION_ROLE_NAME" {
  description = "Execution role name to run SageMaker Processing jobs."
  type        = string
}

variable "MODEL_BUILDING_STEP_FUNCTIONS_NAME" {
  description = "Step Functions name."
  type        = string
}

variable "STEP_FUNCTIONS_EXECUTION_ROLE_NAME" {
  description = "Execution role name to run Step Functions."
  type        = string
}

variable "MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME" {
  description = "Model Building Step Functions file name describing Step Functions steps."
  type        = string
}

# AWS PROVIDER
provider "aws" {
  region = var.REGION_NAME
}

# SAGEMAKER EXECUTION ROLE
resource "aws_iam_role" "sagemaker_execution_role" {
  # SageMaker execution role name
  name = var.SAGEMAKER_EXECUTION_ROLE_NAME
  
  # Tags
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = var.ENV
  }

  # Role policy
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Attach permissions to sagemaker_execution_role
resource "aws_iam_role_policy" "sagemaker_custom_policy" {
  name = "SageMakerCustomPermissions"

  role = aws_iam_role.sagemaker_execution_role.name

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          # S3 full access
          "s3:*",
          "s3-object-lambda:*",
          # SageMaker full access
          "sagemaker:*",
          "sagemaker-geospatial:*",
          # ECR Full access
          "ecr:*",
          # CloudWatch full access
          "logs:*",
          "cloudwatch:*",
          # IAM full access
          "iam:*"
        ],
        Resource = "*"
      }
    ]
  })
}

# STEP FUNCTIONS EXECUTION ROLE
resource "aws_iam_role" "step_functions_execution_role" {
  name = var.STEP_FUNCTIONS_EXECUTION_ROLE_NAME
  
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = var.ENV
  }

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "states.amazonaws.com"
      }
    }]
  })
}

# Attach permissions for Step Functions to create and manage state machines
resource "aws_iam_role_policy" "step_functions_custom_policy" {
  name = "StepFunctionsCustomPermissions"

  role = aws_iam_role.step_functions_execution_role.name

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          # S3 full access
          "s3:*",
          "s3-object-lambda:*",
          # SageMaker full access
          "sagemaker:*",
          "sagemaker-geospatial:*",
          # ECR Full access
          "ecr:*",
          # CloudWatch full access
          "logs:*",
          "cloudwatch:*",
          # Step Functions full access
          "states:*",
          "sts:*",
          # Amazon EventBridge full access
          "events:*",
          # AWS X-Ray full access
          "xray:*",
          # IAM Full access
          "iam:PassRole"
        ],
        Resource = "*"
      }
    ]
  })
}

# MODEL BUILDING WORKFLOW STEP FUNCTION
resource "aws_sfn_state_machine" "model_building_workflow" {
  # Step function name
  name     = var.MODEL_BUILDING_STEP_FUNCTIONS_NAME

  # Role ARN
  role_arn = aws_iam_role.step_functions_execution_role.arn

  # Step Functions definition (loaded from json file)
  definition = file(var.MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME)
}
