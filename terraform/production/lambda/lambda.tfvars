# Production ETL lambda variables

ETL_LAMBDA_FUNCTION_NAME        = "etl-prod"
LAMBDA_EXECUTION_ROLE_NAME      = "LambdaExecutionRole-MLAccelerator-Prod"
ETL_LAMBDA_LOG_GROUP            = "/aws/lambda/etl-prod"
ETL_LAMBDA_FUNCTION_MEMORY_SIZE = 512
ETL_LAMBDA_FUNCTION_TIMEOUT     = 300
SECRET_ARN                      = "arn:aws:secretsmanager:sa-east-1:097866913509:secret:access_keys-DBqbT3"