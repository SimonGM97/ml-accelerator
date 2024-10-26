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

variable "REGION" {
  description = "AWS region where the S3 bucket will be created."
  type        = string
}

variable "STEP_FUNCTIONS_EXECUTION_ROLE_ARN" {
    description = "Execution role ARN to run SageMaker pipelines."
    type        = string
}

variable "BUCKET_NAME" {
  description = "S3 bucket name where training datasets will be stored."
  type        = string
}

# AWS PROVIDER
provider "aws" {
  region = var.REGION
}

# MODEL BUILDING WORKFLOW STEP FUNCTION
resource "aws_sfn_state_machine" "model_building_workflow" {
  # Step function name
  name     = "ModelBuildingWorkflow"

  # Role ARN
  role_arn = var.STEP_FUNCTIONS_EXECUTION_ROLE_ARN

  definition = jsonencode({
    "Comment": "Model Building Workflow",
    "StartAt": "DataProcessingJob",
    "States": {
      "DataProcessingJob": {
        "Type": "Task",
        "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
        "Parameters": {
          "ProcessingJobName": "DataProcessing",
          "ProcessingResources": {
            "ClusterConfig": {
              "InstanceCount": 1,
              "InstanceType": "ml.m5.xlarge",  # Specify instance type
              "VolumeSizeInGB": 30
            }
          },
          "AppSpecification": {
            "ImageUri": "your-docker-image-url",
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/data_processing.py"]
          },
          "RoleArn": "${var.sagemaker_execution_role_arn}",
          "ProcessingInputs": [{
            "InputName": "input-data",
            "S3Input": {
              "S3Uri": "s3://${var.s3_bucket_name}/input-data/",
              "LocalPath": "/opt/ml/processing/input",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File",
              "S3DataDistributionType": "FullyReplicated"
            }
          }],
          "ProcessingOutputConfig": {
            "Outputs": [{
              "OutputName": "output-data",
              "S3Output": {
                "S3Uri": "s3://${var.s3_bucket_name}/output-data/",
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob"
              }
            }]
          }
        },
        "Next": "TuningJob"
      },
      "TuningJob": {
        "Type": "Task",
        "Resource": "arn:aws:states:::sagemaker:createHyperParameterTuningJob.sync",
        "Parameters": {
          "HyperParameterTuningJobName": "TuningJob",
          "HyperParameterTuningJobConfig": {
            "Strategy": "Bayesian",
            "HyperParameterTuningResourceConfig": {
              "InstanceType": "ml.m5.xlarge",
              "InstanceCount": 2
            }
          },
          "TrainingJobDefinition": {
            "AlgorithmSpecification": {
              "TrainingImage": "your-docker-image-url",
              "TrainingInputMode": "File"
            },
            "RoleArn": "${var.sagemaker_execution_role_arn}",
            "InputDataConfig": [{
              "ChannelName": "train",
              "DataSource": {
                "S3DataSource": {
                  "S3Uri": "s3://${var.s3_bucket_name}/train-data/",
                  "S3DataType": "S3Prefix",
                  "S3InputMode": "File"
                }
              }
            }],
            "OutputDataConfig": {
              "S3OutputPath": "s3://${var.s3_bucket_name}/tuning-output/"
            },
            "ResourceConfig": {
              "InstanceType": "ml.m5.large",
              "InstanceCount": 1,
              "VolumeSizeInGB": 50
            }
          }
        },
        "Next": "TrainingJob"
      },
      "TrainingJob": {
        "Type": "Task",
        "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
        "Parameters": {
          "TrainingJobName": "TrainingJob",
          "AlgorithmSpecification": {
            "TrainingImage": "your-docker-image-url",
            "TrainingInputMode": "File"
          },
          "RoleArn": "${var.sagemaker_execution_role_arn}",
          "InputDataConfig": [{
            "ChannelName": "train",
            "DataSource": {
              "S3DataSource": {
                "S3Uri": "s3://${var.s3_bucket_name}/train-data/",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File"
              }
            }
          }],
          "OutputDataConfig": {
            "S3OutputPath": "s3://${var.s3_bucket_name}/training-output/"
          },
          "ResourceConfig": {
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 50
          }
        },
        "Next": "EvaluationJob"
      },
      "EvaluationJob": {
        "Type": "Task",
        "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
        "Parameters": {
          "ProcessingJobName": "EvaluationJob",
          "ProcessingResources": {
            "ClusterConfig": {
              "InstanceCount": 1,
              "InstanceType": "ml.m5.xlarge",
              "VolumeSizeInGB": 30
            }
          },
          "AppSpecification": {
            "ImageUri": "your-docker-image-url",
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/evaluating.py"]
          },
          "RoleArn": "${var.sagemaker_execution_role_arn}",
          "ProcessingInputs": [{
            "InputName": "input-data",
            "S3Input": {
              "S3Uri": "s3://${var.s3_bucket_name}/test-data/",
              "LocalPath": "/opt/ml/processing/input",
              "S3DataType": "S3Prefix",
              "S3InputMode": "File",
              "S3DataDistributionType": "FullyReplicated"
            }
          }],
          "ProcessingOutputConfig": {
            "Outputs": [{
              "OutputName": "evaluation-results",
              "S3Output": {
                "S3Uri": "s3://${var.s3_bucket_name}/evaluation-output/",
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob"
              }
            }]
          }
        },
        "End": true
      }
    }
  })
}
