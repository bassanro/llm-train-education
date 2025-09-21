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

## License

MIT License