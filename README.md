# NAPAL LLM Test Generation System

A comprehensive system for fine-tuning Large Language Models to generate NAPAL (National Assessment of Performance in Literacy) tests for Year 3 students, with automated scoring capabilities.

## Features

- **LLM Fine-tuning**: Custom fine-tuning pipeline for generating age-appropriate test questions
- **Question Generation**: Automated generation of new test questions annually
- **Automated Scoring**: AI-powered scoring system for student responses
- **Evaluation Framework**: Tools to assess question quality and scoring accuracy
- **Annual Workflow**: System to generate fresh question sets each year

## Project Structure

```
napal_llm_system/
├── data/
│   ├── raw/                    # Original NAPAL test data
│   └── processed/              # Processed training data
├── models/                     # Fine-tuned models and checkpoints
├── src/
│   ├── training/               # Fine-tuning scripts
│   ├── generation/             # Question generation pipeline
│   ├── scoring/                # Automated scoring system
│   └── evaluation/             # Evaluation and metrics
├── tests/                      # Unit tests
├── generated_tests/            # Output directory for generated tests
└── configs/                    # Configuration files
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your NAPAL training data in the `data/raw/` directory

3. Run the data preparation pipeline:
```bash
python src/training/prepare_data.py
```

4. Fine-tune the model:
```bash
python src/training/fine_tune.py
```

5. Generate new test questions:
```bash
python src/generation/generate_test.py --year 2025
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model parameters
- Training hyperparameters
- Question generation settings
- Scoring criteria

## Usage

### Generate a New Test
```python
from src.generation.test_generator import NAPALTestGenerator

generator = NAPALTestGenerator()
test = generator.generate_test(year=2025, num_questions=20)
```

### Score Student Responses
```python
from src.scoring.scorer import NAPALScorer

scorer = NAPALScorer()
scores = scorer.score_responses(test_questions, student_answers)
```

## Deploying training to AWS (cheapest practical method)

This project includes helpers in `src/training/fine_tune.py` to upload datasets to S3 and launch a one-time EC2 Spot instance that will run the training script. The approach minimizes cost by using Spot instances and LoRA/4-bit quantization.

High-level steps

1. Add AWS configuration to `configs/config.yaml` (see example below).
2. Ensure `boto3` and `awscli` are installed (they are included in `requirements.txt`).
3. Run locally to upload datasets & launch a Spot instance:

```bash
python src/training/fine_tune.py --aws-deploy --config configs/config.yaml
```

Example `configs/config.yaml` additions

```yaml
aws:
  s3_bucket: "my-napal-bucket"
  s3_prefix: "napal-datasets"
  region: "ap-southeast-2"
  ami_id: "ami-0abcd1234example"        # replace with a GPU DL AMI in your region
  repo_url: "https://github.com/owner/llm-train-education.git"
  spot_instance_type: "g4dn.xlarge"
  spot_max_price: 0.6
  iam_instance_profile: "napal-training-role"
  key_name: "my-ec2-keypair"
```

Minimal IAM policy for the launcher and instance (attach to the user who launches and the instance role respectively)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-napal-bucket",
        "arn:aws:s3:::my-napal-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:DescribeInstances",
        "ec2:TerminateInstances",
        "ec2:RequestSpotInstances",
        "ec2:RequestSpotFleet",
        "ec2:DescribeSpotInstanceRequests"
      ],
      "Resource": "*"
    }
  ]
}
```

Caveats and notes

- The bootstrap user-data assumes the instance can access S3 (via IAM role). If not, you must provide AWS credentials on the instance.
- Choose an AMI with CUDA and the drivers for your chosen instance (Deep Learning AMIs or AWS Deep Learning Containers).
- Spot instances may be interrupted; the script includes a watcher and checkpoint upload to S3 to reduce lost work.

If you want, I can also add a CloudFormation template to create the S3 bucket and instance role automatically.

## License

MIT License