"""
Data preparation pipeline for NAPAL test fine-tuning.
Converts raw NAPAL test data into training format for LLM fine-tuning.
"""

import json
import jsonschema
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data preparation"""
    raw_data_dir: Path
    processed_data_dir: Path
    schema_file: Path
    train_split: float = 0.8
    eval_split: float = 0.2
    max_examples_per_type: int = 1000

class NAPALDataProcessor:
    """Process NAPAL test data for LLM fine-tuning"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict:
        """Load JSON schema for validation"""
        with open(self.config.schema_file, 'r') as f:
            return json.load(f)

    def validate_test_data(self, test_data: Dict) -> bool:
        """Validate test data against schema"""
        try:
            jsonschema.validate(test_data, self.schema)
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def extract_training_examples(self, test_data: Dict) -> List[Dict]:
        """Extract training examples from test data"""
        examples = []

        for question in test_data['questions']:
            # Create training examples for question generation
            examples.extend(self._create_question_examples(question, test_data['test_metadata']))

            # Create training examples for answer generation
            examples.extend(self._create_answer_examples(question))

            # Create training examples for scoring rubric generation
            examples.extend(self._create_rubric_examples(question))

        return examples

    def _create_question_examples(self, question: Dict, metadata: Dict) -> List[Dict]:
        """Create training examples for question generation"""
        examples = []

        # Format the instruction for question generation
        instruction = self._format_question_instruction(question, metadata)

        # Format the expected output
        output = self._format_question_output(question)

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "task_type": "question_generation",
            "subject_area": question["subject_area"],
            "difficulty_level": question["difficulty_level"],
            "question_type": question["question_type"]
        })

        return examples

    def _create_answer_examples(self, question: Dict) -> List[Dict]:
        """Create training examples for answer generation"""
        examples = []

        instruction = f"Generate the correct answer and explanation for this Year 3 NAPAL question:\\n\\n{question['question_text']}"

        if question.get('stimulus') and question['stimulus'].get('content'):
            instruction = f"Based on this text:\\n{question['stimulus']['content']}\\n\\n{instruction}"

        output = self._format_answer_output(question['correct_answer'])

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "task_type": "answer_generation",
            "subject_area": question["subject_area"],
            "question_type": question["question_type"]
        })

        return examples

    def _create_rubric_examples(self, question: Dict) -> List[Dict]:
        """Create training examples for scoring rubric generation"""
        examples = []

        instruction = f"Generate a scoring rubric for this Year 3 NAPAL question:\\n\\n{question['question_text']}"
        output = self._format_rubric_output(question['scoring_rubric'])

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "task_type": "rubric_generation",
            "subject_area": question["subject_area"],
            "question_type": question["question_type"]
        })

        return examples

    def _format_question_instruction(self, question: Dict, metadata: Dict) -> str:
        """Format instruction for question generation"""
        base_instruction = f"""Generate a {question['question_type']} question for Year 3 NAPAL assessment.

Requirements:
- Subject area: {question['subject_area']}
- Difficulty level: {question['difficulty_level']}/5
- Learning objective: {question['learning_objective']}
- Appropriate for 8-9 year old students
"""

        if question.get('stimulus') and question['stimulus'].get('content'):
            base_instruction += f"\\nStimulus text provided: {question['stimulus']['content']}"

        return base_instruction

    def _format_question_output(self, question: Dict) -> str:
        """Format expected output for question generation"""
        output = f"Question: {question['question_text']}\\n"

        if question.get('options'):
            output += "Options:\\n"
            for option in question['options']:
                output += f"{option['option_id']}. {option['text']}\\n"

        return output.strip()

    def _format_answer_output(self, correct_answer: Dict) -> str:
        """Format expected output for answer generation"""
        output = f"Answer: {correct_answer['answer']}\\n"

        if correct_answer.get('explanation'):
            output += f"Explanation: {correct_answer['explanation']}\\n"

        if correct_answer.get('acceptable_variations'):
            output += f"Acceptable variations: {', '.join(correct_answer['acceptable_variations'])}"

        return output.strip()

    def _format_rubric_output(self, rubric: Dict) -> str:
        """Format expected output for rubric generation"""
        output = f"Points possible: {rubric['points_possible']}\\n\\n"
        output += "Scoring criteria:\\n"

        for criteria in rubric.get('scoring_criteria', []):
            output += f"{criteria['points']} points: {criteria['criteria']}\\n"

        return output.strip()

    def process_all_files(self) -> Tuple[List[Dict], List[Dict]]:
        """Process all JSON files in raw data directory"""
        all_examples = []

        for json_file in self.config.raw_data_dir.glob("*.json"):
            logger.info(f"Processing {json_file}")

            with open(json_file, 'r') as f:
                test_data = json.load(f)

            if self.validate_test_data(test_data):
                examples = self.extract_training_examples(test_data)
                all_examples.extend(examples)
                logger.info(f"Extracted {len(examples)} examples from {json_file}")
            else:
                logger.warning(f"Skipping invalid file: {json_file}")

        # Split into train and eval sets
        if not all_examples:
            logger.warning("No training examples extracted from raw data")
            return [], []

        stratify_labels = [ex['task_type'] for ex in all_examples]
        try:
            train_examples, eval_examples = train_test_split(
                all_examples,
                test_size=self.config.eval_split,
                random_state=42,
                stratify=stratify_labels
            )
        except ValueError as e:
            # This can happen with very small datasets where stratification isn't possible
            logger.warning(f"Stratified split failed ({e}); falling back to non-stratified split")
            train_examples, eval_examples = train_test_split(
                all_examples,
                test_size=self.config.eval_split,
                random_state=42
            )

        logger.info(f"Total examples: {len(all_examples)}")
        logger.info(f"Train examples: {len(train_examples)}")
        logger.info(f"Eval examples: {len(eval_examples)}")

        return train_examples, eval_examples

    def save_processed_data(self, train_examples: List[Dict], eval_examples: List[Dict]):
        """Save processed data to JSONL files"""
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Save training data
        train_file = self.config.processed_data_dir / "train_data.jsonl"
        with open(train_file, 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\\n')

        # Save evaluation data
        eval_file = self.config.processed_data_dir / "eval_data.jsonl"
        with open(eval_file, 'w') as f:
            for example in eval_examples:
                f.write(json.dumps(example) + '\\n')

        logger.info(f"Saved training data to {train_file}")
        logger.info(f"Saved evaluation data to {eval_file}")

        # Save data statistics
        self._save_data_stats(train_examples, eval_examples)

    def _save_data_stats(self, train_examples: List[Dict], eval_examples: List[Dict]):
        """Save data statistics for analysis"""
        train_df = pd.DataFrame(train_examples)
        eval_df = pd.DataFrame(eval_examples)

        stats = {
            "total_examples": len(train_examples) + len(eval_examples),
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "task_type_distribution": train_df['task_type'].value_counts().to_dict(),
            "subject_area_distribution": train_df['subject_area'].value_counts().to_dict(),
            "difficulty_distribution": train_df['difficulty_level'].value_counts().to_dict(),
            "question_type_distribution": train_df['question_type'].value_counts().to_dict()
        }

        stats_file = self.config.processed_data_dir / "data_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved data statistics to {stats_file}")

def main():
    """Main function to run data preparation"""
    # Load configuration
    config_path = Path("configs/config.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create data config
    data_config = DataConfig(
        raw_data_dir=Path("data/raw"),
        processed_data_dir=Path("data/processed"),
        schema_file=Path("data/napal_test_schema.json"),
        train_split=0.8,
        eval_split=0.2
    )

    # Initialize processor
    processor = NAPALDataProcessor(data_config)

    # Process data
    train_examples, eval_examples = processor.process_all_files()

    # Save processed data
    processor.save_processed_data(train_examples, eval_examples)

    logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()