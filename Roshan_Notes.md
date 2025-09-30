I added a new helper script: submit_sagemaker_training.py

What it does:

Uploads train/eval files (from configs/config.yaml) to S3 under a prefix.
Creates and submits a SageMaker HuggingFace estimator training job that runs src/training/fine_tune.py.
Accepts CLI overrides for role ARN, region, instance type/count, and job name.
Also added the sagemaker SDK to requirements.txt.

How to use

Install dependencies (preferably in a virtualenv matching Python 3.9+):

pip install -r requirements.txt
Ensure your AWS credentials are configured and you have a SageMaker execution role ARN ready.

Run:

python submit_sagemaker_training.py --config config.yaml --role-arn arn:aws:iam::123456789012:role/SageMakerRole --region ap-southeast-2
Notes and caveats

The SageMaker role must allow S3 access to the bucket configured in config.yaml and SageMaker training permissions.
The script uses the built-in Hugging Face images by default for compatibility; you can override with --image-uri to use a custom ECR image.
The training script expects datasets to be available in the training and eval channels; the estimator is configured to point channels to the S3 prefixes uploaded by this script.
If you want, I can:

Create a CloudFormation template or Terraform configuration to provision the S3 bucket and SageMaker role automatically.
Produce a small test job with a very small dataset to validate end-to-end before running full training.