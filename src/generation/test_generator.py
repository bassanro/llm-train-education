"""
NAPAL Test Generator - Generate new test questions using fine-tuned LLM
"""

import json
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import uuid
import random

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel
import re

from src.training.utils import TextProcessor, QuestionValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NAPALTestGenerator:
    """Generate NAPAL tests using fine-tuned LLM"""

    def __init__(self, config_path: str = "configs/config.yaml", model_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model_path = model_path or self._get_latest_model_path()
        self.text_processor = TextProcessor()
        self.validator = QuestionValidator()

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

        # Question templates and prompts
        self.question_templates = self._load_question_templates()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_latest_model_path(self) -> str:
        """Get path to the latest trained model"""
        models_dir = Path(self.config['training']['output_dir'])

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Look for final_model first, then latest checkpoint
        final_model = models_dir / "final_model"
        if final_model.exists():
            return str(final_model)

        # Find latest checkpoint
        checkpoints = list(models_dir.glob("checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError("No trained models found")

        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
        return str(latest_checkpoint)

    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['base_model'],
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

        logger.info("Model loaded successfully")

    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question generation templates"""
        return {
            "reading_comprehension": [
                "Generate a reading comprehension question for Year 3 students about {topic}. Include a short passage and a multiple choice question with 4 options.",
                "Create a Year 3 reading comprehension question that asks students to identify the main idea of a passage about {topic}.",
                "Generate a reading comprehension question for 8-9 year olds that tests inference skills using a story about {topic}."
            ],
            "vocabulary": [
                "Create a vocabulary question for Year 3 students to test understanding of the word '{word}' using context clues.",
                "Generate a Year 3 vocabulary question that asks students to choose the correct meaning of '{word}' from multiple options.",
                "Create a vocabulary question for 8-9 year olds about synonyms for the word '{word}'."
            ],
            "writing": [
                "Generate a Year 3 writing prompt asking students to write a descriptive paragraph about {topic}.",
                "Create a writing question for Year 3 students that asks them to write a short story beginning with '{prompt_start}'.",
                "Generate a Year 3 writing task about {topic} that requires students to use descriptive language."
            ],
            "language_conventions": [
                "Create a Year 3 grammar question about {grammar_topic} with multiple choice options.",
                "Generate a language conventions question for Year 3 students about punctuation in sentences.",
                "Create a grammar question for 8-9 year olds about {grammar_topic} with practical examples."
            ]
        }

    def generate_question(self,
                         subject_area: str,
                         difficulty_level: int,
                         question_type: str,
                         topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate a single question"""

        # Get appropriate template
        template = self._select_template(subject_area, question_type, topic)

        # Generate question using the model
        generated_content = self._generate_with_model(template, subject_area, difficulty_level)

        # Parse the generated content
        question_data = self._parse_generated_question(generated_content, subject_area, difficulty_level, question_type)

        # Validate the question
        validation_result = self.validator.validate_question(question_data)

        if not validation_result["is_valid"]:
            logger.warning(f"Generated question validation failed: {validation_result['errors']}")
            # Try regenerating once
            generated_content = self._generate_with_model(template, subject_area, difficulty_level)
            question_data = self._parse_generated_question(generated_content, subject_area, difficulty_level, question_type)

        # Add metadata
        question_data.update({
            "question_id": str(uuid.uuid4())[:8],
            "generated_at": datetime.now().isoformat(),
            "validation_score": validation_result["quality_score"],
            "validation_warnings": validation_result.get("warnings", [])
        })

        return question_data

    def _select_template(self, subject_area: str, question_type: str, topic: Optional[str]) -> str:
        """Select appropriate template for question generation"""
        templates = self.question_templates.get(subject_area, [])

        if not templates:
            templates = ["Generate a {question_type} question for Year 3 students in {subject_area}."]

        template = random.choice(templates)

        # Fill in template variables
        if "{topic}" in template:
            if not topic:
                topic = self._generate_topic(subject_area)
            template = template.format(topic=topic)

        if "{word}" in template:
            word = self._select_vocabulary_word(subject_area)
            template = template.format(word=word)

        if "{grammar_topic}" in template:
            grammar_topic = random.choice(["nouns", "verbs", "adjectives", "punctuation", "sentence structure"])
            template = template.format(grammar_topic=grammar_topic)

        if "{prompt_start}" in template:
            prompt_start = random.choice([
                "Once upon a time...",
                "It was a sunny day when...",
                "The mysterious door opened and...",
                "My best friend and I discovered..."
            ])
            template = template.format(prompt_start=prompt_start)

        return template

    def _generate_topic(self, subject_area: str) -> str:
        """Generate age-appropriate topic"""
        topics = {
            "reading_comprehension": [
                "animals", "friendship", "family", "school", "adventures", "nature",
                "sports", "pets", "holidays", "community helpers"
            ],
            "vocabulary": [
                "animals", "emotions", "weather", "food", "transportation",
                "school subjects", "family members", "colors", "shapes"
            ],
            "writing": [
                "my favorite place", "a special day", "an amazing animal",
                "my best friend", "a fun adventure", "my family"
            ],
            "language_conventions": [
                "playground rules", "classroom activities", "weekend plans",
                "favorite foods", "pet care"
            ]
        }

        return random.choice(topics.get(subject_area, topics["reading_comprehension"]))

    def _select_vocabulary_word(self, subject_area: str) -> str:
        """Select age-appropriate vocabulary word"""
        words = [
            "brave", "curious", "gentle", "excited", "worried", "proud",
            "enormous", "tiny", "swift", "careful", "friendly", "clever"
        ]
        return random.choice(words)

    def _generate_with_model(self, prompt: str, subject_area: str, difficulty_level: int) -> str:
        """Generate content using the fine-tuned model"""

        # Format prompt for instruction tuning
        formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

Requirements:
- Subject area: {subject_area}
- Difficulty level: {difficulty_level}/5
- Appropriate for Year 3 students (ages 8-9)
- Include correct answer and explanation

### Response:"""

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=500,
                temperature=self.config['model']['temperature'],
                top_p=self.config['model']['top_p'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text

        return response

    def _parse_generated_question(self, generated_content: str, subject_area: str, difficulty_level: int, question_type: str) -> Dict[str, Any]:
        """Parse generated content into structured question format"""

        # Initialize question structure
        question_data = {
            "question_type": question_type,
            "subject_area": subject_area,
            "difficulty_level": difficulty_level,
            "question_text": "",
            "correct_answer": {"answer": "", "explanation": ""},
            "scoring_rubric": {"points_possible": 1, "scoring_criteria": []},
            "tags": []
        }

        # Parse different sections
        lines = generated_content.split('\\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Identify sections
            if line.lower().startswith(('question:', 'q:')):
                current_section = 'question'
                question_data['question_text'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith(('answer:', 'a:')):
                current_section = 'answer'
                question_data['correct_answer']['answer'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('explanation:'):
                current_section = 'explanation'
                question_data['correct_answer']['explanation'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('options:'):
                current_section = 'options'
                question_data['options'] = []
            elif current_section == 'options' and re.match(r'^[A-D][\\.\\)]', line):
                option_id = line[0]
                option_text = line[2:].strip()
                if 'options' not in question_data:
                    question_data['options'] = []
                question_data['options'].append({
                    "option_id": option_id,
                    "text": option_text
                })
            elif current_section == 'question' and not question_data['question_text']:
                question_data['question_text'] = line
            elif current_section == 'answer' and not question_data['correct_answer']['answer']:
                question_data['correct_answer']['answer'] = line
            elif current_section == 'explanation' and not question_data['correct_answer']['explanation']:
                question_data['correct_answer']['explanation'] = line

        # Set appropriate scoring rubric based on question type
        if question_type == "multiple_choice":
            question_data['scoring_rubric'] = {
                "points_possible": 1,
                "scoring_criteria": [
                    {"points": 1, "criteria": "Correct answer selected"},
                    {"points": 0, "criteria": "Incorrect answer selected"}
                ]
            }
        elif question_type == "short_answer":
            question_data['scoring_rubric'] = {
                "points_possible": 2,
                "scoring_criteria": [
                    {"points": 2, "criteria": "Complete and accurate answer"},
                    {"points": 1, "criteria": "Partially correct answer"},
                    {"points": 0, "criteria": "Incorrect or no answer"}
                ]
            }
        else:  # extended_response
            question_data['scoring_rubric'] = {
                "points_possible": 4,
                "scoring_criteria": [
                    {"points": 4, "criteria": "Excellent response with clear ideas and good organization"},
                    {"points": 3, "criteria": "Good response with mostly clear ideas"},
                    {"points": 2, "criteria": "Adequate response with some clear ideas"},
                    {"points": 1, "criteria": "Basic response with limited clarity"},
                    {"points": 0, "criteria": "Inadequate or no response"}
                ]
            }

        # Add appropriate tags
        question_data['tags'] = [subject_area, question_type, f"difficulty_{difficulty_level}"]

        return question_data

    def generate_test(self,
                     year: int,
                     num_questions: Optional[int] = None,
                     custom_distribution: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a complete NAPAL test"""

        if num_questions is None:
            num_questions = self.config['generation']['num_questions_per_test']

        # Use custom distribution or default from config
        if custom_distribution:
            subject_dist = custom_distribution.get('subject_distribution', self.config['generation']['subject_distribution'])
            difficulty_dist = custom_distribution.get('difficulty_distribution', self.config['generation']['difficulty_distribution'])
            type_dist = custom_distribution.get('question_type_distribution', self.config['generation']['question_type_distribution'])
        else:
            subject_dist = self.config['generation']['subject_distribution']
            difficulty_dist = self.config['generation']['difficulty_distribution']
            type_dist = self.config['generation']['question_type_distribution']

        # Calculate number of questions for each category
        question_plan = self._create_question_plan(num_questions, subject_dist, difficulty_dist, type_dist)

        logger.info(f"Generating {num_questions} questions for Year {year}")
        logger.info(f"Question plan: {question_plan}")

        # Generate questions
        questions = []
        for plan_item in question_plan:
            question = self.generate_question(
                subject_area=plan_item['subject_area'],
                difficulty_level=plan_item['difficulty_level'],
                question_type=plan_item['question_type']
            )
            questions.append(question)

        # Create test metadata
        test_metadata = {
            "test_id": f"NAPAL_Y3_{year}_{datetime.now().strftime('%m%d_%H%M')}",
            "year": year,
            "grade_level": "Year 3",
            "subject_areas": list(set(q['subject_area'] for q in questions)),
            "total_questions": len(questions),
            "estimated_duration_minutes": self._estimate_duration(questions),
            "created_date": datetime.now().isoformat(),
            "generation_config": {
                "subject_distribution": subject_dist,
                "difficulty_distribution": difficulty_dist,
                "question_type_distribution": type_dist
            }
        }

        # Compile test
        test_data = {
            "test_metadata": test_metadata,
            "questions": questions
        }

        logger.info(f"Generated test with {len(questions)} questions")

        return test_data

    def _create_question_plan(self, num_questions: int, subject_dist: Dict, difficulty_dist: Dict, type_dist: Dict) -> List[Dict]:
        """Create a plan for question generation"""
        plan = []

        # Calculate distributions
        subjects = [(k, int(v * num_questions)) for k, v in subject_dist.items()]
        difficulties = [(k.split('_')[1], int(v * num_questions)) for k, v in difficulty_dist.items()]
        types = [(k, int(v * num_questions)) for k, v in type_dist.items()]

        # Adjust for rounding
        while sum(count for _, count in subjects) < num_questions:
            subjects[0] = (subjects[0][0], subjects[0][1] + 1)

        # Create combinations
        subject_list = []
        for subject, count in subjects:
            subject_list.extend([subject] * count)

        difficulty_list = []
        for difficulty, count in difficulties:
            difficulty_list.extend([int(difficulty)] * count)

        type_list = []
        for q_type, count in types:
            type_list.extend([q_type] * count)

        # Shuffle and combine
        random.shuffle(subject_list)
        random.shuffle(difficulty_list)
        random.shuffle(type_list)

        for i in range(num_questions):
            plan.append({
                'subject_area': subject_list[i % len(subject_list)],
                'difficulty_level': difficulty_list[i % len(difficulty_list)],
                'question_type': type_list[i % len(type_list)]
            })

        return plan

    def _estimate_duration(self, questions: List[Dict]) -> int:
        """Estimate test duration in minutes"""
        base_times = {
            "multiple_choice": 1.5,
            "short_answer": 2.5,
            "extended_response": 5.0,
            "cloze": 2.0,
            "matching": 2.0
        }

        total_minutes = sum(base_times.get(q['question_type'], 2.0) for q in questions)
        return int(total_minutes) + 5  # Add 5 minutes buffer

    def save_test(self, test_data: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save generated test to file"""
        if output_dir is None:
            output_dir = self.config['paths']['output_dir']

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename
        test_id = test_data['test_metadata']['test_id']
        filename = f"{test_id}.json"
        filepath = output_path / filename

        # Save test
        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)

        logger.info(f"Test saved to {filepath}")
        return str(filepath)

def main():
    """Main function for generating tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate NAPAL test")
    parser.add_argument("--year", type=int, default=2025, help="Year for the test")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of questions")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")

    args = parser.parse_args()

    # Initialize generator
    generator = NAPALTestGenerator(config_path=args.config)

    # Generate test
    test_data = generator.generate_test(
        year=args.year,
        num_questions=args.num_questions
    )

    # Save test
    filepath = generator.save_test(test_data, args.output_dir)

    print(f"Generated NAPAL test for Year {args.year}")
    print(f"Test saved to: {filepath}")
    print(f"Number of questions: {test_data['test_metadata']['total_questions']}")
    print(f"Estimated duration: {test_data['test_metadata']['estimated_duration_minutes']} minutes")

if __name__ == "__main__":
    main()