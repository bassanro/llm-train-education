"""Submit a SageMaker HuggingFace training job for this repo.

Uploads datasets from `configs/config.yaml` to S3 (bucket/prefix), then creates and starts
a SageMaker HuggingFace estimator that runs `src/training/fine_tune.py` as the entry_point.

Usage (example):

python scripts/submit_sagemaker_training.py --config configs/config.yaml --role-arn arn:aws:iam::123456789012:role/SageMakerRole --region ap-southeast-2

The script reads aws settings from the config but allows overrides via CLI.
"""
import argparse
import os
import sys
import time
import yaml
from pathlib import Path
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def upload_datasets_to_s3(cfg: dict, s3_client, bucket: str, prefix: str) -> dict:
    """Upload train/eval files as s3://{bucket}/{prefix}/train/... and /eval/...

    Returns dict with keys 'train' and 'eval' pointing to their S3 URIs.
    """
    data_cfg = cfg['data']
    train_local = Path(data_cfg['train_file'])
    eval_local = Path(data_cfg['eval_file'])

    base = prefix.strip('/').rstrip('/') if prefix else 'napal-datasets'
    train_prefix = f"{base}/train"
    eval_prefix = f"{base}/eval"

    s3_paths = {}

    for local_path, target_prefix, key in [
        (train_local, train_prefix, 'train'),
        (eval_local, eval_prefix, 'eval')
    ]:
        if not local_path.exists():
            print(f"Warning: {local_path} does not exist; skipping upload for {key}")
            continue
        s3_key = f"{target_prefix}/{local_path.name}"
        print(f"Uploading {local_path} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(local_path), bucket, s3_key)
        s3_paths[key] = f"s3://{bucket}/{target_prefix}"

    return s3_paths


def submit_job(config_path: str, role_arn: str, region: str, instance_type: str, instance_count: int, image_uri: str = None, job_name: str = None):
    cfg = load_config(config_path)
    aws_cfg = cfg.get('aws', {})

    bucket = aws_cfg.get('s3_bucket')
    prefix = aws_cfg.get('s3_prefix', 'napal-datasets')

    if not bucket:
        print('Error: s3_bucket must be configured in configs/config.yaml under aws.s3_bucket or provided in the config. Aborting.')
        sys.exit(1)

    boto_sess = boto3.Session(region_name=region)
    s3_client = boto_sess.client('s3')

    # Upload datasets
    s3_paths = upload_datasets_to_s3(cfg, s3_client, bucket, prefix)
    if not s3_paths:
        print('No datasets uploaded; aborting.')
        sys.exit(1)

    # Build estimator hyperparameters
    hf_hyperparams = {
        'config_path': 'configs/config.yaml',
        'train_file': s3_paths.get('train'),
        'eval_file': s3_paths.get('eval')
    }

    sess = sagemaker.Session(boto_session=boto_sess)

    entry_point = aws_cfg.get('sagemaker_entry_point', 'src/training/fine_tune.py')
    source_dir = aws_cfg.get('sagemaker_source_dir', '.')
    # Use a SageMaker-supported transformers version by default
    transformers_version = aws_cfg.get('transformers_version', '4.36.0')
    # Use a compatible pytorch version; adjust if you have specific needs
    pytorch_version = aws_cfg.get('pytorch_version', '2.1.1')
    py_version = aws_cfg.get('py_version', 'py39')

    estimator_kwargs = dict(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role_arn,
        instance_type=instance_type,
        instance_count=instance_count,
        transformers_version=transformers_version,
        pytorch_version=pytorch_version,
        py_version=py_version,
        hyperparameters=hf_hyperparams,
        sagemaker_session=sess
    )

    if image_uri:
        estimator_kwargs['image_uri'] = image_uri

    print('Creating HuggingFace estimator with the following args:')
    for k, v in estimator_kwargs.items():
        if k == 'sagemaker_session':
            continue
        print(f'  {k}: {v}')

    huggingface_estimator = HuggingFace(**estimator_kwargs)

    data_channels = {}
    if 'train' in s3_paths:
        data_channels['train'] = s3_paths['train']
    if 'eval' in s3_paths:
        data_channels['eval'] = s3_paths['eval']

    if not job_name:
        job_name = f"napal-hf-{int(time.time())}"

    print(f"Submitting SageMaker training job '{job_name}' ...")
    huggingface_estimator.fit(inputs=data_channels, job_name=job_name)
    print('Job submitted. Check SageMaker console for progress.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit SageMaker HuggingFace training job')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
    parser.add_argument('--role-arn', type=str, required=False, help='IAM role ARN to use for SageMaker (overrides config)')
    parser.add_argument('--region', type=str, default='ap-southeast-2')
    parser.add_argument('--instance-type', type=str, default=None, help='SageMaker instance type (default from config or ml.g4dn.xlarge)')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--image-uri', type=str, default=None, help='Optional ECR image uri to use instead of built-in HuggingFace images')
    parser.add_argument('--job-name', type=str, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    aws_cfg = cfg.get('aws', {})

    role = args.role_arn or aws_cfg.get('sagemaker_role_arn') or aws_cfg.get('role_arn')
    if not role:
        print('Error: An IAM role ARN for SageMaker is required. Provide via --role-arn or configs/config.yaml (aws.sagemaker_role_arn).')
        sys.exit(1)

    region = args.region or aws_cfg.get('region', 'ap-southeast-2')
    instance_type = args.instance_type or aws_cfg.get('sagemaker_instance_type', 'ml.g4dn.xlarge')

    submit_job(args.config, role, region, instance_type, args.instance_count, image_uri=args.image_uri, job_name=args.job_name)
