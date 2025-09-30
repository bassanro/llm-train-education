#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <bucket-name> <role-name> [region]"
  exit 1
fi

BUCKET_NAME="$1"
ROLE_NAME="$2"
REGION="${3:-ap-southeast-2}"

STACK_NAME="napal-sagemaker-stack-${BUCKET_NAME}"
TEMPLATE_FILE="$(pwd)/infra/sagemaker_stack.yaml"

echo "Creating CloudFormation stack $STACK_NAME in region $REGION..."
aws cloudformation deploy \
  --template-file "$TEMPLATE_FILE" \
  --stack-name "$STACK_NAME" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides BucketName="$BUCKET_NAME" RoleName="$ROLE_NAME" \
  --region "$REGION"

echo "Stack deployment complete. Fetching outputs..."
aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" --query 'Stacks[0].Outputs' --output table

echo "Done. The S3 bucket and SageMaker role should be provisioned."
