"""
Automated scoring system for NAPAL test responses.
Uses semantic similarity, rule-based matching, and rubric-based evaluation.
"""

import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import difflib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NAPALScorer:
    """Automated scorer for NAPAL test responses"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.scoring_config = self.config.get('scoring', {})

        # Initialize NLP tools
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features may not work.")
            self.nlp = None

        # Initialize vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def score_response(self, question: Dict[str, Any], student_response: str) -> Dict[str, Any]:
        """Score a single student response"""

        question_type = question.get('question_type', '')
        correct_answer = question.get('correct_answer', {})
        scoring_rubric = question.get('scoring_rubric', {})

        # Route to appropriate scoring method
        if question_type == 'multiple_choice':
            return self._score_multiple_choice(question, student_response)
        elif question_type == 'short_answer':
            return self._score_short_answer(question, student_response)
        elif question_type == 'extended_response':
            return self._score_extended_response(question, student_response)
        elif question_type == 'cloze':
            return self._score_cloze(question, student_response)
        else:
            return self._score_generic(question, student_response)

    def _score_multiple_choice(self, question: Dict, student_response: str) -> Dict[str, Any]:
        """Score multiple choice question"""
        correct_answer = question['correct_answer']['answer'].strip().upper()
        student_answer = student_response.strip().upper()

        # Extract single letter if response contains extra text
        if len(student_answer) > 1:
            match = re.search(r'[A-D]', student_answer)
            if match:
                student_answer = match.group()
            else:
                student_answer = student_answer[0] if student_answer else ''

        # Check if correct
        is_correct = student_answer == correct_answer
        points = question['scoring_rubric']['points_possible'] if is_correct else 0

        result = {
            'points_earned': points,
            'points_possible': question['scoring_rubric']['points_possible'],
            'is_correct': is_correct,
            'student_answer': student_answer,
            'correct_answer': correct_answer,
            'feedback': self._generate_mc_feedback(question, student_answer, is_correct),
            'confidence': 1.0  # High confidence for exact matches
        }

        return result

    def _score_short_answer(self, question: Dict, student_response: str) -> Dict[str, Any]:
        """Score short answer question"""
        correct_answer = question['correct_answer']['answer']
        acceptable_variations = question['correct_answer'].get('acceptable_variations', [])

        # Normalize responses
        student_normalized = self._normalize_text(student_response)
        correct_normalized = self._normalize_text(correct_answer)

        # Check exact match first
        if student_normalized == correct_normalized:
            return self._create_score_result(question, 2, True, 1.0, "Exact match with correct answer")

        # Check acceptable variations
        for variation in acceptable_variations:
            if student_normalized == self._normalize_text(variation):
                return self._create_score_result(question, 2, True, 0.95, "Matches acceptable variation")

        # Calculate semantic similarity
        similarity_score = self._calculate_semantic_similarity(student_response, correct_answer)

        # Determine points based on similarity and rubric
        if similarity_score >= self.scoring_config.get('similarity_threshold', 0.8):
            points = 2
            is_correct = True
            feedback = "Response demonstrates good understanding"
        elif similarity_score >= 0.6:
            points = 1
            is_correct = False
            feedback = "Response shows partial understanding"
        else:
            points = 0
            is_correct = False
            feedback = "Response does not demonstrate understanding"

        return self._create_score_result(question, points, is_correct, similarity_score, feedback)

    def _score_extended_response(self, question: Dict, student_response: str) -> Dict[str, Any]:
        """Score extended response question"""
        scoring_criteria = question['scoring_rubric'].get('scoring_criteria', [])

        # Analyze response components
        analysis = self._analyze_extended_response(student_response, question)

        # Determine score based on rubric
        points = self._apply_extended_rubric(analysis, scoring_criteria)
        points_possible = question['scoring_rubric']['points_possible']

        is_correct = points >= (points_possible * 0.6)  # 60% threshold for "correct"

        feedback = self._generate_extended_feedback(analysis, points, points_possible)

        return {
            'points_earned': points,
            'points_possible': points_possible,
            'is_correct': is_correct,
            'feedback': feedback,
            'analysis': analysis,
            'confidence': 0.8  # Lower confidence for subjective scoring
        }

    def _score_cloze(self, question: Dict, student_response: str) -> Dict[str, Any]:
        """Score cloze (fill-in-the-blank) question"""
        # Similar to short answer but may have multiple blanks
        return self._score_short_answer(question, student_response)

    def _score_generic(self, question: Dict, student_response: str) -> Dict[str, Any]:
        """Generic scoring for unknown question types"""
        logger.warning(f"Unknown question type: {question.get('question_type')}")
        return self._score_short_answer(question, student_response)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower().strip()

        # Remove punctuation
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)

        # Remove stop words and stem
        words = word_tokenize(text)
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]

        return ' '.join(words)

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            # Use TF-IDF vectorization
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Also check string similarity as backup
            string_similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

            # Use the higher of the two scores
            return max(similarity, string_similarity)

        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            # Fallback to string similarity
            return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _analyze_extended_response(self, response: str, question: Dict) -> Dict[str, Any]:
        """Analyze extended response for various quality metrics"""
        if not response.strip():
            return {
                'word_count': 0,
                'sentence_count': 0,
                'has_content': False,
                'organization_score': 0,
                'content_score': 0,
                'language_quality': 0
            }

        # Basic metrics
        words = word_tokenize(response)
        sentences = sent_tokenize(response)

        # Content analysis
        has_relevant_content = self._check_content_relevance(response, question)

        # Organization analysis
        organization_score = self._analyze_organization(response)

        # Language quality
        language_quality = self._analyze_language_quality(response)

        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'has_content': has_relevant_content,
            'organization_score': organization_score,
            'content_score': 0.8 if has_relevant_content else 0.2,
            'language_quality': language_quality
        }

    def _check_content_relevance(self, response: str, question: Dict) -> bool:
        """Check if response is relevant to the question"""
        question_text = question.get('question_text', '')

        # Extract key concepts from question
        if self.nlp:
            question_doc = self.nlp(question_text)
            question_concepts = [token.lemma_.lower() for token in question_doc
                               if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop]
        else:
            question_concepts = [word.lower() for word in word_tokenize(question_text)
                               if word.lower() not in self.stop_words]

        # Check if response contains relevant concepts
        response_lower = response.lower()
        relevant_concepts = sum(1 for concept in question_concepts if concept in response_lower)

        return relevant_concepts >= max(1, len(question_concepts) * 0.3)

    def _analyze_organization(self, response: str) -> float:
        """Analyze organization of extended response"""
        sentences = sent_tokenize(response)

        if len(sentences) < 2:
            return 0.3

        # Check for transition words/phrases
        transitions = ['first', 'second', 'then', 'next', 'finally', 'also', 'however', 'because', 'therefore']
        has_transitions = any(transition in response.lower() for transition in transitions)

        # Check for consistent topic (very basic)
        organization_score = 0.5
        if has_transitions:
            organization_score += 0.3
        if len(sentences) >= 3:
            organization_score += 0.2

        return min(1.0, organization_score)

    def _analyze_language_quality(self, response: str) -> float:
        """Analyze language quality of response"""
        # Basic checks for Year 3 level
        words = word_tokenize(response)
        sentences = sent_tokenize(response)

        if not words:
            return 0.0

        # Check average sentence length (appropriate for Year 3)
        avg_sentence_length = len(words) / len(sentences)
        length_score = 0.8 if 5 <= avg_sentence_length <= 15 else 0.5

        # Check for spelling/grammar (basic)
        # This is simplified - in practice, you'd use more sophisticated tools
        uppercase_ratio = sum(1 for c in response if c.isupper()) / len(response)
        capitalization_score = 0.8 if uppercase_ratio < 0.3 else 0.5

        # Check for complete sentences
        has_periods = response.count('.') >= len(sentences) * 0.7
        punctuation_score = 0.8 if has_periods else 0.5

        return (length_score + capitalization_score + punctuation_score) / 3

    def _apply_extended_rubric(self, analysis: Dict, scoring_criteria: List[Dict]) -> int:
        """Apply rubric to determine points for extended response"""
        if not analysis['has_content']:
            return 0

        # Calculate overall quality score
        content_weight = self.scoring_config.get('rubric_weights', {}).get('content_accuracy', 0.6)
        language_weight = self.scoring_config.get('rubric_weights', {}).get('language_quality', 0.3)
        completeness_weight = self.scoring_config.get('rubric_weights', {}).get('completeness', 0.1)

        # Determine completeness based on word count
        completeness_score = min(1.0, analysis['word_count'] / 30)  # 30 words = complete for Year 3

        overall_score = (
            analysis['content_score'] * content_weight +
            analysis['language_quality'] * language_weight +
            completeness_score * completeness_weight
        )

        # Map to rubric points
        if overall_score >= 0.85:
            return max([criteria['points'] for criteria in scoring_criteria])
        elif overall_score >= 0.70:
            return max([criteria['points'] for criteria in scoring_criteria]) - 1
        elif overall_score >= 0.50:
            return max([criteria['points'] for criteria in scoring_criteria]) - 2
        elif overall_score >= 0.30:
            return 1
        else:
            return 0

    def _create_score_result(self, question: Dict, points: int, is_correct: bool, confidence: float, feedback: str) -> Dict[str, Any]:
        """Create standardized score result"""
        return {
            'points_earned': points,
            'points_possible': question['scoring_rubric']['points_possible'],
            'is_correct': is_correct,
            'feedback': feedback,
            'confidence': confidence
        }

    def _generate_mc_feedback(self, question: Dict, student_answer: str, is_correct: bool) -> str:
        """Generate feedback for multiple choice questions"""
        if is_correct:
            return "Correct! Well done."
        else:
            correct_answer = question['correct_answer']['answer']
            explanation = question['correct_answer'].get('explanation', '')

            feedback = f"The correct answer is {correct_answer}."
            if explanation:
                feedback += f" {explanation}"

            return feedback

    def _generate_extended_feedback(self, analysis: Dict, points: int, points_possible: int) -> str:
        """Generate feedback for extended responses"""
        feedback_parts = []

        if points == points_possible:
            feedback_parts.append("Excellent response!")
        elif points >= points_possible * 0.75:
            feedback_parts.append("Good response with clear ideas.")
        elif points >= points_possible * 0.5:
            feedback_parts.append("Adequate response that shows understanding.")
        else:
            feedback_parts.append("Your response needs more development.")

        # Specific suggestions
        if analysis['word_count'] < 20:
            feedback_parts.append("Try to write more to fully explain your ideas.")

        if analysis['language_quality'] < 0.6:
            feedback_parts.append("Check your spelling and punctuation.")

        if not analysis['has_content']:
            feedback_parts.append("Make sure your answer relates to the question.")

        return " ".join(feedback_parts)

    def score_test(self, test_data: Dict, student_responses: Dict[str, str]) -> Dict[str, Any]:
        """Score an entire test"""
        results = {
            'test_metadata': test_data['test_metadata'],
            'total_points_possible': 0,
            'total_points_earned': 0,
            'percentage_score': 0,
            'question_scores': [],
            'summary': {},
            'scored_at': datetime.now().isoformat()
        }

        for question in test_data['questions']:
            question_id = question['question_id']
            student_response = student_responses.get(question_id, "")

            # Score the question
            score_result = self.score_response(question, student_response)

            # Add question context
            score_result.update({
                'question_id': question_id,
                'question_text': question['question_text'],
                'subject_area': question['subject_area'],
                'question_type': question['question_type'],
                'student_response': student_response
            })

            results['question_scores'].append(score_result)
            results['total_points_possible'] += score_result['points_possible']
            results['total_points_earned'] += score_result['points_earned']

        # Calculate percentage
        if results['total_points_possible'] > 0:
            results['percentage_score'] = (results['total_points_earned'] / results['total_points_possible']) * 100

        # Generate summary
        results['summary'] = self._generate_test_summary(results)

        return results

    def _generate_test_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary of test performance"""
        question_scores = results['question_scores']

        # Performance by subject area
        subject_performance = {}
        for score in question_scores:
            subject = score['subject_area']
            if subject not in subject_performance:
                subject_performance[subject] = {'correct': 0, 'total': 0, 'points_earned': 0, 'points_possible': 0}

            subject_performance[subject]['total'] += 1
            subject_performance[subject]['points_earned'] += score['points_earned']
            subject_performance[subject]['points_possible'] += score['points_possible']
            if score['is_correct']:
                subject_performance[subject]['correct'] += 1

        # Calculate percentages
        for subject_data in subject_performance.values():
            if subject_data['points_possible'] > 0:
                subject_data['percentage'] = (subject_data['points_earned'] / subject_data['points_possible']) * 100
            else:
                subject_data['percentage'] = 0

        # Performance by question type
        type_performance = {}
        for score in question_scores:
            q_type = score['question_type']
            if q_type not in type_performance:
                type_performance[q_type] = {'correct': 0, 'total': 0}

            type_performance[q_type]['total'] += 1
            if score['is_correct']:
                type_performance[q_type]['correct'] += 1

        # Overall grade
        percentage = results['percentage_score']
        if percentage >= 90:
            grade = 'A'
        elif percentage >= 80:
            grade = 'B'
        elif percentage >= 70:
            grade = 'C'
        elif percentage >= 60:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'overall_grade': grade,
            'subject_performance': subject_performance,
            'question_type_performance': type_performance,
            'recommendations': self._generate_recommendations(subject_performance, percentage)
        }

    def _generate_recommendations(self, subject_performance: Dict, overall_percentage: float) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []

        # Overall performance
        if overall_percentage < 60:
            recommendations.append("Continue practicing reading and literacy skills.")
        elif overall_percentage < 80:
            recommendations.append("Good progress! Focus on areas that need improvement.")
        else:
            recommendations.append("Excellent work! Keep up the great effort.")

        # Subject-specific recommendations
        for subject, performance in subject_performance.items():
            if performance['percentage'] < 60:
                if subject == 'reading_comprehension':
                    recommendations.append("Practice reading comprehension with short stories and passages.")
                elif subject == 'vocabulary':
                    recommendations.append("Work on expanding vocabulary through reading and word games.")
                elif subject == 'writing':
                    recommendations.append("Practice descriptive writing and organizing ideas in paragraphs.")
                elif subject == 'language_conventions':
                    recommendations.append("Review grammar, punctuation, and sentence structure.")

        return recommendations

def main():
    """Main function for testing the scorer"""
    # Example usage
    scorer = NAPALScorer()

    # Load a test
    test_file = "generated_tests/sample_test.json"
    if Path(test_file).exists():
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        # Sample student responses
        student_responses = {
            "Q001": "B",
            "Q002": "breakable",
            "Q003": "My favorite place is the park. It has green grass and tall trees. I can hear birds singing and children playing. The air smells fresh and clean."
        }

        # Score the test
        results = scorer.score_test(test_data, student_responses)

        print(f"Test Score: {results['percentage_score']:.1f}%")
        print(f"Grade: {results['summary']['overall_grade']}")
        print("\\nQuestion Scores:")
        for score in results['question_scores']:
            print(f"  {score['question_id']}: {score['points_earned']}/{score['points_possible']} points")

if __name__ == "__main__":
    main()