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
  description = "AWS region where the resources will be created."
  type        = string
}

variable "SAGEMAKER_EXECUTION_ROLE_NAME" {
  description = "Execution role name to run SageMaker Processing jobs (on Dev environment)."
  type        = string
}

variable "MODEL_BUILDING_STEP_FUNCTIONS_NAME" {
  description = "Step Functions name (on Dev environment)."
  type        = string
}

variable "STEP_FUNCTIONS_EXECUTION_ROLE_NAME" {
  description = "Execution role name to run Step Functions (on Dev environment)."
  type        = string
}

variable "MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME" {
  description = "Model Building Step Functions file name describing Step Functions steps (on Dev environment)."
  type        = string
}

# AWS PROVIDER
provider "aws" {
  region = var.REGION_NAME
}

# SAGEMAKER EXECUTION ROLE
resource "aws_iam_role" "sagemaker_execution_role_dev" {
  # SageMaker execution role name
  name = var.SAGEMAKER_EXECUTION_ROLE_NAME
  
  # Tags
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = "dev"
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

# Attach permissions to sagemaker_execution_role_dev
resource "aws_iam_role_policy" "sagemaker_custom_policy_dev" {
  name = "SageMakerCustomPermissionsDev"

  role = aws_iam_role.sagemaker_execution_role_dev.name

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
resource "aws_iam_role" "step_functions_execution_role_dev" {
  name = var.STEP_FUNCTIONS_EXECUTION_ROLE_NAME
  
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = "dev"
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
resource "aws_iam_role_policy" "step_functions_custom_policy_dev" {
  name = "StepFunctionsCustomPermissionsDev"

  role = aws_iam_role.step_functions_execution_role_dev.name

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
resource "aws_sfn_state_machine" "model_building_workflow_dev" {
  # Step function name
  name     = var.MODEL_BUILDING_STEP_FUNCTIONS_NAME

  # Role ARN
  role_arn = aws_iam_role.step_functions_execution_role_dev.arn

  # Step Functions definition (loaded from json file)
  definition = file(var.MODEL_BUILDING_STEP_FUNCTIONS_FILE_NAME)

  # Tags
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = "dev"
  }

  # Specify logging configuration
  # logging_configuration {
  #   log_destination        = "${aws_cloudwatch_log_group.log_group_for_sfn.arn}:*"
  #   include_execution_data = true
  #   level                  = "INFO"
  # }
}
