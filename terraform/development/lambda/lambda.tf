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

variable "ETL_LAMBDA_FUNCTION_NAME" {
  description = "ETL lambda function name (on Dev environment)."
  type        = string
}

variable "LAMBDA_EXECUTION_ROLE_NAME" {
  description = "Execution role name to run Lambda functions (on Dev environment)."
  type        = string
}

variable "ETL_LAMBDA_IMAGE_URI" {
  description = "Docker image that will be ran by the ETL lambda function (on Dev environment)."
  type        = string
}

variable "ETL_LAMBDA_LOG_GROUP" {
  description = "Log group where the logg messages will be saved (on Dev environment)."
  type        = string
} 

variable "ETL_LAMBDA_FUNCTION_MEMORY_SIZE" {
  description = "Memory assigned to the lambda function (on Dev environment)."
  type        = string
}

variable "ETL_LAMBDA_FUNCTION_TIMEOUT" {
  description = "Amount of time in seconds that the lambda function is allowed to run (on Dev environment)."
  type        = string
}

# AWS PROVIDER
provider "aws" {
  # Region name
  region = var.REGION_NAME
}

# LAMBDA EXECUTION ROLE
resource "aws_iam_role" "lambda_execution_role_dev" {
  # Lambda execution role name
  name = var.LAMBDA_EXECUTION_ROLE_NAME

  # Tags
  tags = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = "dev"
  }
  
  # Role definition
  assume_role_policy = jsonencode({
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
          "Service": "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Attach permissions to lambda_execution_role_dev
resource "aws_iam_role_policy" "sagemaker_custom_policy_dev" {
  name = "SageMakerCustomPermissionsProd"

  role = aws_iam_role.lambda_execution_role_dev.name

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          # S3 full access
          "s3:*",
          "s3-object-lambda:*",
          # ECR Full access
          "ecr:*",
          # CloudWatch full access
          "logs:*",
          "cloudwatch:*",
          # Secrets Manager
          "secretsmanager:GetSecretValue",
          "secretsmanager:PutSecretValue"
        ],
        Resource = "*"
      }
    ]
  })
}

# LAMBDA FUNCTION
resource "aws_lambda_function" "etl_lambda_dev" {
  # Lambda function name
  function_name                  = var.ETL_LAMBDA_FUNCTION_NAME

  # Lambda function description
  description                    = "Lambda function that will run the etl.py script."
  
  # Execution role
  role                           = aws_iam_role.lambda_execution_role_dev.arn

  # Lambda deployment package type
  package_type                   = "Image"

  # ECR image uri
  image_uri                      = var.ETL_LAMBDA_IMAGE_URI
  
  # Architercure
  architectures                  = ["x86_64"]

  # Environment variables
  # environment -> Already defined in the Docker image

  # Tags
  tags                           = {
    Project     = var.PROJECT_NAME
    Version     = var.VERSION
    Environment = "dev"
  }

  # Logging configuration
  logging_config {
    log_format            = "JSON"
    application_log_level = "DEBUG"
    system_log_level      = "DEBUG"
    log_group             = var.ETL_LAMBDA_LOG_GROUP
  }

  # Lambda memory size
  memory_size                    = var.ETL_LAMBDA_FUNCTION_MEMORY_SIZE

  #  The amount of Ephemeral storage (/tmp) to allocate for the Lambda Function in MB.
  ephemeral_storage {
    size = 512
  }
  
  # Amount of time your Lambda Function has to run in seconds.
  timeout                        = var.ETL_LAMBDA_FUNCTION_TIMEOUT

  # Amount of reserved concurrent executions for this lambda function. 
  #     - A value of 0 disables lambda from being triggered and -1 removes any concurrency limitations. 
  #     - Defaults to Unreserved Concurrency Limits -1
  reserved_concurrent_executions = -1

  # Whether to publish creation/change as new Lambda Function Version.
  publish                        = false
}