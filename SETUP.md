# NAPAL LLM System Setup Guide

## Prerequisites

### Installing Python and pip

#### Windows

1. **Check if Python is already installed:**
   ```cmd
   python --version
   ```
   If Python 3.8+ is installed, check if pip is available:
   ```cmd
   pip --version
   ```

2. **If Python is not installed:**
   - Download Python from [python.org](https://www.python.org/downloads/)
   - **Important:** Check "Add Python to PATH" during installation
   - pip is included automatically with Python 3.4+

3. **If Python is installed but pip is missing:**
   ```cmd
   python -m ensurepip --upgrade
   ```

4. **Alternative pip installation methods:**
   - Download [get-pip.py](https://bootstrap.pypa.io/get-pip.py)
   - Run: `python get-pip.py`

#### macOS

1. **Using Homebrew (recommended):**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install Python (includes pip)
   brew install python
   ```

2. **Using the official installer:**
   - Download from [python.org](https://www.python.org/downloads/)
   - pip is included automatically

3. **If pip is missing:**
   ```bash
   python3 -m ensurepip --upgrade
   ```

#### Linux (Ubuntu/Debian)

1. **Install Python and pip:**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. **For other distributions:**
   ```bash
   # CentOS/RHEL/Fedora
   sudo yum install python3 python3-pip
   # or
   sudo dnf install python3 python3-pip

   # Arch Linux
   sudo pacman -S python python-pip
   ```

#### Verify Installation

After installation, verify both Python and pip work:
```bash
python --version    # Should show Python 3.8+
pip --version      # Should show pip version
```

**Note for Windows users:** You might need to use `python` and `pip` or `python3` and `pip3` depending on your installation.

#### Virtual Environment (Recommended)

Create an isolated environment for this project:
```bash
# Create virtual environment
python -m venv napal_env

# Activate it
# Windows:
napal_env\Scripts\activate
# macOS/Linux:
source napal_env/bin/activate

# Your prompt should now show (napal_env)
```

## Quick Start

### 1. Install Dependencies
```bash
cd napal_llm_system

# Make sure pip is up to date
python -m pip install --upgrade pip


# Install project dependencies
pip install -r requirements.txt
```

**If you encounter issues:**
- **Permission errors on Windows:** Try `pip install --user -r requirements.txt`
- **Permission errors on macOS/Linux:** Try `pip3 install --user -r requirements.txt`
- **Network issues:** Try `pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org`
- **Dependency conflicts:** Use virtual environment (see Prerequisites section)

**For GPU support (optional but recommended):**
```bash
# Install PyTorch with CUDA support (check pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download Required Models
```bash
# Download spaCy model for text processing
python -m spacy download en_core_web_sm
```

### 3. Run Demo
```bash
python demo.py
```

### 4. Use the System

#### Generate Tests
```bash
# Generate test for 2025 with 20 questions
python main.py generate --year 2025 --num-questions 20

# Generate annual test suite (3 variants)
python main.py workflow --generate 2025 --variants 3
```

#### Score Student Responses
```bash
python main.py score --test-file generated_tests/test.json --responses student_responses.json
```

#### Evaluate Test Quality
```bash
python main.py evaluate --test-file generated_tests/test.json
```

## Full Workflow

### Step 1: Prepare Training Data
1. Add your NAPAL test data to `data/raw/` (JSON format following the schema)
2. Run data preparation:
```bash
python main.py prepare-data
```

### Step 2: Fine-tune the Model
```bash
python main.py fine-tune
```

### Step 3: Generate Tests
```bash
# Single test
python main.py generate --year 2025

# Annual workflow with multiple variants
python main.py workflow --generate 2025
```

### Step 4: Use Generated Tests
- Tests are saved in `generated_tests/`
- Use the scoring system to evaluate student responses
- Use evaluation tools to assess question quality

## Configuration

Edit `configs/config.yaml` to customize:
- Model parameters
- Training settings
- Question generation preferences
- Scoring criteria
- Quality thresholds

## Directory Structure

```
napal_llm_system/
├── data/
│   ├── raw/                    # Original training data
│   ├── processed/              # Processed training data
│   └── napal_test_schema.json  # Data schema
├── models/                     # Fine-tuned models
├── src/
│   ├── training/               # Data prep & fine-tuning
│   ├── generation/             # Question generation
│   ├── scoring/                # Automated scoring
│   ├── evaluation/             # Quality evaluation
│   └── workflow/               # Annual management
├── generated_tests/            # Generated test outputs
├── configs/                    # Configuration files
├── tests/                      # Unit tests
├── main.py                     # CLI interface
├── demo.py                     # Demo script
└── requirements.txt            # Dependencies
```

## Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **"ModuleNotFoundError: No module named 'spacy'"**

   This means the spaCy package is not installed in the active Python environment. Common reasons: you installed dependencies into a different Python interpreter, or you didn't activate your virtual environment.

   Recommended fixes (macOS / Linux zsh):
   ```bash
   # If using a virtual environment, activate it first
   source napal_env/bin/activate

   # Upgrade pip and install spaCy
   python -m pip install --upgrade pip
   python -m pip install spacy

   # Install the small English model
   python -m spacy download en_core_web_sm
   ```

   If you prefer to install all project dependencies at once, run:
   ```bash
   pip install -r requirements.txt
   ```

   If you still see the error, verify which python and pip are active:
   ```bash
   which python
   which pip
   python -c "import sys; print(sys.executable)"
   python -c "import importlib; print(importlib.util.find_spec('spacy') is not None)"
   ```

3. **"No trained model found"**
   - You need to fine-tune a model first
   - Or download a pre-trained NAPAL model if available

4. **CUDA out of memory**
   - Reduce batch size in `configs/config.yaml`
   - Use CPU training by setting `fp16: false`

5. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path if running from different directories

### Performance Tips

1. **For faster training:**
   - Use GPU if available
   - Increase batch size
   - Use mixed precision (fp16: true)

2. **For better quality:**
   - Add more training data
   - Increase training epochs
   - Tune hyperparameters

3. **For production use:**
   - Set up automated annual generation
   - Implement human review workflow
   - Monitor quality metrics over time

## API Usage

You can also use the system programmatically:

```python
from src.generation.test_generator import NAPALTestGenerator
from src.scoring.scorer import NAPALScorer

# Generate test
generator = NAPALTestGenerator()
test = generator.generate_test(year=2025, num_questions=20)

# Score responses
scorer = NAPALScorer()
results = scorer.score_test(test, student_responses)
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the demo.py file for examples
3. Check the configuration in configs/config.yaml
4. Review the system logs for detailed error messages