"""
Fine-tuning script for NAPAL test generation using LoRA and transformers.
"""

import os
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime
import argparse

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb
from accelerate import Accelerator
import wandb
import threading
import time
import urllib.request
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model and training"""
    base_model: str
    model_name: str
    max_length: int
    temperature: float
    top_p: float

@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str
    task_type: str

class NAPALTrainer:
    """NAPAL-specific trainer for LLM fine-tuning"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model_config = ModelConfig(**self.config['model'])
        self.lora_config = LoRAConfig(**self.config['lora'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize accelerator
        self.accelerator = Accelerator()

        # Setup logging
        self._setup_logging()

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        # AWS-related settings (optional)
        aws_cfg = self.config.get('aws', {})
        self.use_spot = bool(aws_cfg.get('use_spot', False))
        self.s3_bucket = aws_cfg.get('s3_bucket')
        self.s3_prefix = aws_cfg.get('s3_prefix', '')
        self._spot_thread = None

        # If running inside SageMaker training container, adapt dataset paths
        self._adapt_for_sagemaker()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging and wandb"""
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['wandb']['project'],
                name=f"napal-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )

    def load_tokenizer_and_model(self):
        """Load tokenizer and model"""
        logger.info(f"Loading tokenizer and model: {self.model_config.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=True
        )

        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with 4-bit quantization for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)

        # Setup LoRA
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(model, peft_config)

        # Print trainable parameters
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_param = 0

        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )

    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training"""
        logger.info(f"Loading dataset from {data_path}")

        # Load JSONL data
        examples = []
        with open(data_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        # Format examples for instruction tuning
        formatted_examples = []
        for example in examples:
            formatted_text = self._format_example(example)
            formatted_examples.append({"text": formatted_text})

        dataset = Dataset.from_list(formatted_examples)

        # Tokenize dataset
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.model_config.max_length,
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def _format_example(self, example: Dict) -> str:
        """Format training example using Alpaca-style instruction format"""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        if input_text:
            formatted = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            formatted = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

        return formatted

    def train(self):
        """Run the training process"""
        logger.info("Starting training process")

        # Load datasets
        train_dataset = self.prepare_dataset(self.config['data']['train_file'])
        eval_dataset = self.prepare_dataset(self.config['data']['eval_file'])

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=self.config['training']['save_total_limit'],
            remove_unused_columns=False,
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            fp16=self.config['training']['fp16'],
            report_to="wandb" if self.config.get('wandb', {}).get('enabled', False) else None,
            run_name=f"napal-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # If running on AWS Spot, start a watcher thread that monitors termination notices
        if self.use_spot:
            logger.info("AWS Spot mode enabled — starting spot interruption watcher thread")
            self._spot_thread = threading.Thread(target=self._spot_watcher, args=(trainer,), daemon=True)
            self._spot_thread.start()

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        final_model_path = Path(self.config['training']['output_dir']) / "final_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        # Upload final model to S3 if configured
        if self.s3_bucket:
            try:
                logger.info(f"Uploading final model to s3://{self.s3_bucket}/{self.s3_prefix}")
                self._upload_dir_to_s3(final_model_path, self.s3_bucket, self.s3_prefix)
            except Exception as e:
                logger.warning(f"Failed to upload final model to S3: {e}")

        logger.info(f"Training completed. Model saved to {final_model_path}")

        # Save training metrics
        self._save_training_metrics(trainer)

    def _save_training_metrics(self, trainer):
        """Save training metrics and logs"""
        metrics_dir = Path(self.config['training']['output_dir']) / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # Save training history
        if hasattr(trainer.state, 'log_history'):
            with open(metrics_dir / "training_history.json", 'w') as f:
                json.dump(trainer.state.log_history, f, indent=2)

        # Save final metrics
        final_metrics = {
            "final_train_loss": trainer.state.log_history[-1].get("train_loss", "N/A"),
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A"),
            "total_training_time": trainer.state.log_history[-1].get("train_runtime", "N/A"),
            "best_model_checkpoint": trainer.state.best_model_checkpoint
        }

        with open(metrics_dir / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)

        logger.info("Training metrics saved")

    def _upload_dir_to_s3(self, local_dir: Path, bucket: str, prefix: str = ""):
        """Upload directory contents to S3 under the given bucket/prefix."""
        s3 = boto3.client('s3')
        local_dir = Path(local_dir)
        prefix = prefix.strip('/') if prefix else ''

        for root, _, files in os.walk(local_dir):
            for fname in files:
                full_path = Path(root) / fname
                relative_path = full_path.relative_to(local_dir)
                s3_key = f"{prefix}/{relative_path}" if prefix else str(relative_path)
                try:
                    s3.upload_file(str(full_path), bucket, s3_key)
                except (BotoCoreError, ClientError) as e:
                    logger.warning(f"Failed to upload {full_path} to s3://{bucket}/{s3_key}: {e}")

    def upload_dataset_files_to_s3(self) -> Dict[str, str]:
        """Upload train/eval dataset files to S3 and return their S3 URIs.

        Expects aws.s3_bucket and aws.s3_prefix (optional) in config.
        """
        aws_cfg = self.config.get('aws', {})
        bucket = aws_cfg.get('s3_bucket') or self.s3_bucket
        prefix = aws_cfg.get('s3_prefix') or self.s3_prefix or 'datasets'

        if not bucket:
            logger.warning("No S3 bucket configured; cannot upload datasets.")
            return {}

        files = [self.config['data']['train_file'], self.config['data']['eval_file']]
        s3_paths = {}
        for fp in files:
            local = Path(fp)
            if not local.exists():
                logger.warning(f"Dataset file {local} not found, skipping upload.")
                continue
            key = f"{prefix}/{local.name}"
            try:
                boto3.client('s3').upload_file(str(local), bucket, key)
                s3_uri = f"s3://{bucket}/{key}"
                s3_paths[local.name] = s3_uri
                logger.info(f"Uploaded {local} -> {s3_uri}")
            except Exception as e:
                logger.warning(f"Failed to upload {local} to S3: {e}")

        return s3_paths

    def _build_user_data_script(self) -> str:
        """Build user-data (bash) script to bootstrap an EC2 spot instance to run this training script.

        Expects the following keys under config['aws'] when using --aws-deploy:
         - repo_url: repository URL (https or ssh) that contains this project
         - s3_bucket / s3_prefix: where datasets were uploaded
         - python_cmd (optional): python binary to use (default: python3)
        """
        aws_cfg = self.config.get('aws', {})
        repo_url = aws_cfg.get('repo_url', '')
        bucket = aws_cfg.get('s3_bucket', self.s3_bucket)
        prefix = aws_cfg.get('s3_prefix', self.s3_prefix)
        python_cmd = aws_cfg.get('python_cmd', 'python3')

        # Train and eval filenames (relative names)
        train_name = Path(self.config['data']['train_file']).name
        eval_name = Path(self.config['data']['eval_file']).name

        # Use a simple bootstrap that assumes the instance has an IAM role allowing S3 and
        # access to Secrets Manager / Systems Manager if needed. The script clones the repo
        # (if repo_url provided) or simply downloads datasets from S3 and runs the training.
        user_data = f'''#!/bin/bash
set -e
# Basic bootstrap (Amazon Linux 2 / Ubuntu compatible commands)
apt-get update || yum update -y || true
# Install git, python3 and pip if missing
if ! command -v git >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get install -y git python3-venv python3-pip
  else
    yum install -y git python3 python3-venv python3-pip
  fi
fi

cd /home/ubuntu || cd /home/ec2-user || cd /root || cd /tmp
# Clone repository if provided
if [ -n "{repo_url}" ]; then
  if [ -d app ]; then
    cd app && git pull || true
  else
    git clone {repo_url} app || true
    cd app || true
  fi
else
  mkdir -p app && cd app
fi

# Create virtualenv and install requirements
{python_cmd} -m venv venv
source venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt --no-cache-dir || true
fi

# Download datasets from S3 (requires instance role / AWS creds)
if [ -n "{bucket}" ]; then
  aws s3 cp s3://{bucket}/{prefix}/{train_name} ./data/{train_name} || true
  aws s3 cp s3://{bucket}/{prefix}/{eval_name} ./data/{eval_name} || true
  # Update config to point to local downloaded dataset
  sed -i "s|{self.config['data']['train_file']}|data/{train_name}|g" configs/config.yaml || true
  sed -i "s|{self.config['data']['eval_file']}|data/{eval_name}|g" configs/config.yaml || true
fi

# Run training (do not attempt to re-launch AWS deploy)
{python_cmd} src/training/fine_tune.py --no-aws-deploy
'''
        return user_data

    def launch_spot_training_instance(self):
        """Launch a cheap EC2 Spot instance to run the training script.

        The function expects certain AWS fields inside config['aws']:
         - ami_id (required)
         - key_name (optional)
         - security_group_ids (optional list)
         - iam_instance_profile (optional name)
         - spot_instance_type (defaults to g4dn.xlarge)
         - spot_max_price (optional)
        """
        aws_cfg = self.config.get('aws', {})
        region = aws_cfg.get('region')
        ami = aws_cfg.get('ami_id')
        if not ami:
            logger.error('AMI ID must be specified in config["aws"]["ami_id"] to launch instances')
            return

        instance_type = aws_cfg.get('spot_instance_type', 'g4dn.xlarge')
        max_price = str(aws_cfg.get('spot_max_price', '0.5'))
        key_name = aws_cfg.get('key_name')
        security_group_ids = aws_cfg.get('security_group_ids', [])
        iam_profile = aws_cfg.get('iam_instance_profile')

        ec2 = boto3.client('ec2', region_name=region) if region else boto3.client('ec2')

        user_data = self._build_user_data_script()

        run_kwargs = dict(
            ImageId=ami,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=user_data
        )
        if key_name:
            run_kwargs['KeyName'] = key_name
        if security_group_ids:
            run_kwargs['SecurityGroupIds'] = security_group_ids
        if iam_profile:
            # Allow both Name or Arn; boto3 accepts dict with Name or Arn key depending on environment
            run_kwargs['IamInstanceProfile'] = {'Name': iam_profile}

        # Request a spot instance using InstanceMarketOptions
        run_kwargs['InstanceMarketOptions'] = {
            'MarketType': 'spot',
            'SpotOptions': {
                'MaxPrice': max_price,
                'SpotInstanceType': 'one-time'
            }
        }

        try:
            resp = ec2.run_instances(**run_kwargs)
            inst_id = resp['Instances'][0]['InstanceId']
            logger.info(f"Launched spot instance {inst_id} (type={instance_type}, ami={ami})")
            return inst_id
        except Exception as e:
            logger.error(f"Failed to launch spot instance: {e}")
            return None

    def _adapt_for_sagemaker(self):
        """If executed inside a SageMaker Training container, adapt config paths to SM channels.

        SageMaker exposes training channel data under /opt/ml/input/data/<channel_name>.
        If environment variables like SM_CHANNEL_TRAIN or SM_CHANNEL_EVAL are present we use
        those paths for train/eval files so the same script can be used as an entry_point.
        """
        sm_train = os.environ.get('SM_CHANNEL_TRAIN') or os.environ.get('SM_CHANNEL_TRAINING')
        sm_eval = os.environ.get('SM_CHANNEL_EVAL') or os.environ.get('SM_CHANNEL_VALIDATION')

        # If channels present, override configured paths to point to the files with the same names
        if sm_train:
            train_fname = Path(self.config['data']['train_file']).name
            new_train = str(Path(sm_train) / train_fname)
            logger.info(f"Detected SageMaker training channel, overriding train_file -> {new_train}")
            self.config['data']['train_file'] = new_train

        if sm_eval:
            eval_fname = Path(self.config['data']['eval_file']).name
            new_eval = str(Path(sm_eval) / eval_fname)
            logger.info(f"Detected SageMaker eval channel, overriding eval_file -> {new_eval}")
            self.config['data']['eval_file'] = new_eval

    def print_sagemaker_run_snippet(self):
        """Upload datasets to S3 and print a SageMaker (HuggingFace estimator) snippet to run the training job.

        This helper does NOT create the job; it uploads the datasets and prints a ready-to-run Python snippet
        that uses the sagemaker SDK. Provide the printed snippet in a separate environment with AWS credentials.
        """
        aws_cfg = self.config.get('aws', {})
        bucket = aws_cfg.get('s3_bucket') or self.s3_bucket
        prefix = aws_cfg.get('s3_prefix') or self.s3_prefix or 'napal-datasets'
        role = aws_cfg.get('sagemaker_role_arn') or aws_cfg.get('role_arn')
        region = aws_cfg.get('region', os.environ.get('AWS_REGION', 'ap-southeast-2'))
        instance_type = aws_cfg.get('sagemaker_instance_type', 'ml.g4dn.xlarge')
        instance_count = int(aws_cfg.get('sagemaker_instance_count', 1))
        entry_point = aws_cfg.get('sagemaker_entry_point', 'src/training/fine_tune.py')
        source_dir = aws_cfg.get('sagemaker_source_dir', '.')
        image_uri = aws_cfg.get('sagemaker_image_uri') or aws_cfg.get('image_uri')

        if not bucket:
            logger.error('S3 bucket must be configured in configs/config.yaml under aws.s3_bucket to prepare SageMaker job snippet')
            return

        s3_paths = self.upload_dataset_files_to_s3()
        if not s3_paths:
            logger.error('Failed to upload datasets to S3; cannot produce SageMaker snippet')
            return

        train_s3 = s3_paths.get(Path(self.config['data']['train_file']).name)
        eval_s3 = s3_paths.get(Path(self.config['data']['eval_file']).name)

        logger.info('Datasets uploaded to S3:')
        logger.info(f'  train: {train_s3}')
        logger.info(f'  eval:  {eval_s3}')

        # Print a SageMaker Python snippet using the sagemaker.huggingface.HuggingFace estimator
        snippet = f"""
# Run this snippet in an environment with the sagemaker SDK installed and AWS credentials configured.
import sagemaker
from sagemaker.huggingface import HuggingFace

sess = sagemaker.Session()
role = '{role}'  # Replace with a role ARN if not set above

hyperparameters = {{
    'train_file': '{train_s3}',
    'eval_file': '{eval_s3}',
    'config_path': 'configs/config.yaml'
}}

# Specify the Hugging Face estimator
huggingface_estimator = HuggingFace(
    entry_point='{entry_point}',
    source_dir='{source_dir}',
    role=role,
    instance_type='{instance_type}',
    instance_count={instance_count},
    transformers_version='4.35.0',
    pytorch_version='2.0.1',
    py_version='py39',
    hyperparameters=hyperparameters,
    sagemaker_session=sess
)

huggingface_estimator.fit({{
    'train': '{train_s3.rsplit('/', 1)[0]}',
    'eval': '{eval_s3.rsplit('/', 1)[0]}'
}})
"""

        logger.info('\nSageMaker run snippet (copy and run in your environment):\n')
        print(snippet)

def main():
    """Main training function with AWS deploy helpers"""
    parser = argparse.ArgumentParser(description='NAPAL fine-tune trainer')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config yaml')
    parser.add_argument('--aws-deploy', action='store_true', help='Upload datasets and launch a cheap AWS Spot instance to run training')
    parser.add_argument('--s3-upload-only', action='store_true', help='Upload datasets to S3 and exit')
    parser.add_argument('--no-aws-deploy', action='store_true', help='When running on remote, avoid trying to re-launch AWS instance')
    parser.add_argument('--sagemaker-snippet', action='store_true', help='Upload datasets to S3 and print a SageMaker/HuggingFace SDK snippet to run the job')
    args = parser.parse_args()

    config_path = args.config

    # Initialize trainer
    trainer = NAPALTrainer(config_path)

    # If requested, upload datasets and launch a spot instance for cheap training
    if args.aws_deploy:
        s3_paths = trainer.upload_dataset_files_to_s3()
        if not s3_paths:
            logger.error('Failed to upload datasets to S3; aborting AWS deploy')
            return
        inst_id = trainer.launch_spot_training_instance()
        if inst_id:
            logger.info(f'AWS spot instance launched: {inst_id}. Training will run on the instance.')
        else:
            logger.error('Failed to launch spot instance')
        return

    if args.s3_upload_only:
        trainer.upload_dataset_files_to_s3()
        return

    if args.sagemaker_snippet:
        trainer.print_sagemaker_run_snippet()
        return

    # Normal local flow: load model/tokenizer and train
    trainer.load_tokenizer_and_model()
    trainer.train()


if __name__ == '__main__':
    main()