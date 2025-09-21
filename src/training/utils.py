"""
Utility functions for data processing and model training.
"""

import json
import re
import string
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextProcessor:
    """Text processing utilities for NAPAL data"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Some features may not work.")
            self.nlp = None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)

        # Remove special characters but keep punctuation for readability
        text = re.sub(r'[^\\w\\s\\.,!?;:()-]', '', text)

        return text.strip()

    def calculate_readability_level(self, text: str) -> float:
        """Calculate Flesch-Kincaid grade level"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        if len(sentences) == 0 or len(words) == 0:
            return 0.0

        # Count syllables
        syllables = sum(self._count_syllables(word) for word in words)

        # Flesch-Kincaid Grade Level formula
        grade_level = (0.39 * (len(words) / len(sentences))) + (11.8 * (syllables / len(words))) - 15.59

        return max(0, grade_level)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using NLP"""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        concepts = []

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'PLACE', 'ORG', 'EVENT']:
                concepts.append(ent.text)

        # Extract important nouns and adjectives
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and
                not token.is_stop and
                not token.is_punct and
                len(token.text) > 2):
                concepts.append(token.lemma_)

        return list(set(concepts))

    def check_age_appropriateness(self, text: str) -> Dict[str, Any]:
        """Check if text is appropriate for Year 3 students"""
        readability = self.calculate_readability_level(text)
        word_count = len(word_tokenize(text))
        sentence_count = len(sent_tokenize(text))
        avg_sentence_length = word_count / max(1, sentence_count)

        # Year 3 appropriate ranges
        appropriate_readability = 2.0 <= readability <= 4.0
        appropriate_sentence_length = avg_sentence_length <= 15
        appropriate_length = word_count <= 200

        return {
            "readability_level": readability,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "is_appropriate": all([
                appropriate_readability,
                appropriate_sentence_length,
                appropriate_length
            ]),
            "recommendations": self._get_recommendations(
                readability, avg_sentence_length, word_count
            )
        }

    def _get_recommendations(self, readability: float, avg_sentence_length: float, word_count: int) -> List[str]:
        """Get recommendations for improving text appropriateness"""
        recommendations = []

        if readability > 4.0:
            recommendations.append("Text is too complex for Year 3. Use simpler words and shorter sentences.")
        elif readability < 2.0:
            recommendations.append("Text might be too simple. Consider adding some complexity.")

        if avg_sentence_length > 15:
            recommendations.append("Sentences are too long. Break them into shorter sentences.")

        if word_count > 200:
            recommendations.append("Text is too long. Consider shortening for Year 3 attention span.")

        return recommendations

class QuestionValidator:
    """Validate NAPAL questions for quality and appropriateness"""

    def __init__(self):
        self.text_processor = TextProcessor()

    def validate_question(self, question: Dict) -> Dict[str, Any]:
        """Validate a single question"""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_score": 0.0
        }

        # Check required fields
        required_fields = ['question_text', 'correct_answer', 'scoring_rubric']
        for field in required_fields:
            if not question.get(field):
                results["errors"].append(f"Missing required field: {field}")
                results["is_valid"] = False

        if not results["is_valid"]:
            return results

        # Check question text appropriateness
        question_analysis = self.text_processor.check_age_appropriateness(question['question_text'])
        if not question_analysis["is_appropriate"]:
            results["warnings"].extend(question_analysis["recommendations"])

        # Check if stimulus is appropriate (if present)
        if question.get('stimulus') and question['stimulus'].get('content'):
            stimulus_analysis = self.text_processor.check_age_appropriateness(question['stimulus']['content'])
            if not stimulus_analysis["is_appropriate"]:
                results["warnings"].extend([f"Stimulus: {rec}" for rec in stimulus_analysis["recommendations"]])

        # Validate multiple choice options
        if question.get('question_type') == 'multiple_choice':
            if not question.get('options') or len(question['options']) < 2:
                results["errors"].append("Multiple choice questions must have at least 2 options")
                results["is_valid"] = False

        # Calculate quality score
        results["quality_score"] = self._calculate_quality_score(question, question_analysis)

        return results

    def _calculate_quality_score(self, question: Dict, analysis: Dict) -> float:
        """Calculate quality score for a question"""
        score = 1.0

        # Penalize for inappropriate readability
        if not analysis["is_appropriate"]:
            score -= 0.3

        # Reward for good structure
        if question.get('learning_objective'):
            score += 0.1

        if question.get('tags'):
            score += 0.1

        # Penalize for missing explanations
        if not question.get('correct_answer', {}).get('explanation'):
            score -= 0.2

        return max(0.0, min(1.0, score))

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file"""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\\n')