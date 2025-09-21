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

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        final_model_path = Path(self.config['training']['output_dir']) / "final_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

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

def main():
    """Main training function"""
    config_path = "configs/config.yaml"

    # Initialize trainer
    trainer = NAPALTrainer(config_path)

    # Load model and tokenizer
    trainer.load_tokenizer_and_model()

    # Start training
    trainer.train()

    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()