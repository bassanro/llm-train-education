"""
Evaluation framework for NAPAL test generation and scoring system.
Assesses question quality, difficulty appropriateness, and scoring accuracy.
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import statistics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from ..training.utils import TextProcessor, QuestionValidator
from ..scoring.scorer import NAPALScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NAPALEvaluator:
    """Comprehensive evaluation framework for NAPAL system"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.eval_config = self.config.get('evaluation', {})

        # Initialize components
        self.text_processor = TextProcessor()
        self.question_validator = QuestionValidator()
        self.scorer = NAPALScorer(config_path)

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def evaluate_generated_questions(self, generated_test: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality of generated questions"""
        logger.info("Evaluating generated questions...")

        questions = generated_test['questions']
        evaluation_results = {
            'test_metadata': generated_test['test_metadata'],
            'total_questions': len(questions),
            'question_evaluations': [],
            'summary_metrics': {},
            'overall_quality_score': 0.0,
            'recommendations': []
        }

        quality_scores = []
        readability_scores = []
        appropriateness_scores = []

        for question in questions:
            question_eval = self._evaluate_single_question(question)
            evaluation_results['question_evaluations'].append(question_eval)

            quality_scores.append(question_eval['quality_score'])
            readability_scores.append(question_eval['readability_metrics']['readability_level'])
            appropriateness_scores.append(question_eval['appropriateness_score'])

        # Calculate summary metrics
        evaluation_results['summary_metrics'] = {
            'average_quality_score': np.mean(quality_scores),
            'quality_score_std': np.std(quality_scores),
            'average_readability': np.mean(readability_scores),
            'readability_appropriateness_rate': sum(1 for score in readability_scores
                                                  if 2.0 <= score <= 4.0) / len(readability_scores),
            'average_appropriateness': np.mean(appropriateness_scores),
            'high_quality_questions': sum(1 for score in quality_scores if score >= 0.8),
            'low_quality_questions': sum(1 for score in quality_scores if score < 0.6),
            'distribution_analysis': self._analyze_question_distribution(questions)
        }

        evaluation_results['overall_quality_score'] = np.mean(quality_scores)
        evaluation_results['recommendations'] = self._generate_quality_recommendations(evaluation_results)

        return evaluation_results

    def _evaluate_single_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question"""

        # Basic validation
        validation_result = self.question_validator.validate_question(question)

        # Readability analysis
        question_text = question.get('question_text', '')
        readability_metrics = self.text_processor.check_age_appropriateness(question_text)

        # Stimulus analysis (if present)
        stimulus_metrics = {}
        if question.get('stimulus') and question['stimulus'].get('content'):
            stimulus_metrics = self.text_processor.check_age_appropriateness(question['stimulus']['content'])

        # Content appropriateness
        appropriateness_score = self._assess_content_appropriateness(question)

        # Difficulty alignment
        difficulty_assessment = self._assess_difficulty_alignment(question)

        # Scoring rubric quality
        rubric_quality = self._assess_rubric_quality(question.get('scoring_rubric', {}))

        return {
            'question_id': question.get('question_id', 'unknown'),
            'question_type': question.get('question_type'),
            'subject_area': question.get('subject_area'),
            'stated_difficulty': question.get('difficulty_level'),
            'validation_result': validation_result,
            'readability_metrics': readability_metrics,
            'stimulus_metrics': stimulus_metrics,
            'appropriateness_score': appropriateness_score,
            'difficulty_assessment': difficulty_assessment,
            'rubric_quality': rubric_quality,
            'quality_score': self._calculate_overall_quality_score(
                validation_result, readability_metrics, appropriateness_score,
                difficulty_assessment, rubric_quality
            )
        }

    def _assess_content_appropriateness(self, question: Dict[str, Any]) -> float:
        """Assess if content is appropriate for Year 3 students"""
        score = 1.0

        question_text = question.get('question_text', '').lower()

        # Check for inappropriate topics
        inappropriate_topics = ['violence', 'adult', 'complex politics', 'death', 'scary']
        for topic in inappropriate_topics:
            if topic in question_text:
                score -= 0.3

        # Check for age-appropriate topics
        appropriate_topics = ['family', 'school', 'animals', 'nature', 'friends', 'games', 'food']
        has_appropriate_topic = any(topic in question_text for topic in appropriate_topics)
        if has_appropriate_topic:
            score += 0.2

        # Check vocabulary complexity
        vocab_analysis = self.text_processor.check_age_appropriateness(question_text)
        if not vocab_analysis['is_appropriate']:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _assess_difficulty_alignment(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if actual difficulty matches stated difficulty"""
        stated_difficulty = question.get('difficulty_level', 3)

        # Analyze various difficulty indicators
        question_text = question.get('question_text', '')

        # Text complexity
        readability = self.text_processor.calculate_readability_level(question_text)

        # Question complexity indicators
        complexity_indicators = {
            'multi_step': 'and' in question_text.lower() or 'also' in question_text.lower(),
            'inference_required': any(word in question_text.lower()
                                    for word in ['why', 'how', 'explain', 'infer', 'suggest']),
            'comparison_required': any(word in question_text.lower()
                                     for word in ['compare', 'contrast', 'similar', 'different']),
            'analysis_required': any(word in question_text.lower()
                                   for word in ['analyze', 'evaluate', 'judge', 'critique'])
        }

        # Estimate actual difficulty
        estimated_difficulty = self._estimate_difficulty(readability, complexity_indicators, question)

        alignment_score = 1.0 - abs(stated_difficulty - estimated_difficulty) / 5.0

        return {
            'stated_difficulty': stated_difficulty,
            'estimated_difficulty': estimated_difficulty,
            'readability_level': readability,
            'complexity_indicators': complexity_indicators,
            'alignment_score': alignment_score,
            'is_well_aligned': alignment_score >= 0.7
        }

    def _estimate_difficulty(self, readability: float, complexity_indicators: Dict, question: Dict) -> float:
        """Estimate actual difficulty level"""
        # Base difficulty from readability
        if readability <= 2.0:
            base_difficulty = 1
        elif readability <= 3.0:
            base_difficulty = 2
        elif readability <= 4.0:
            base_difficulty = 3
        elif readability <= 5.0:
            base_difficulty = 4
        else:
            base_difficulty = 5

        # Adjust for complexity
        complexity_adjustment = sum(complexity_indicators.values()) * 0.5

        # Adjust for question type
        question_type = question.get('question_type', '')
        type_adjustments = {
            'multiple_choice': 0,
            'short_answer': 0.5,
            'extended_response': 1.0,
            'cloze': 0.3
        }

        type_adjustment = type_adjustments.get(question_type, 0.5)

        estimated = base_difficulty + complexity_adjustment + type_adjustment
        return min(5.0, max(1.0, estimated))

    def _assess_rubric_quality(self, rubric: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of scoring rubric"""
        if not rubric:
            return {'quality_score': 0.0, 'issues': ['Missing rubric']}

        issues = []
        quality_score = 1.0

        # Check required fields
        if 'points_possible' not in rubric:
            issues.append('Missing points_possible')
            quality_score -= 0.3

        if 'scoring_criteria' not in rubric:
            issues.append('Missing scoring_criteria')
            quality_score -= 0.5
        else:
            criteria = rubric['scoring_criteria']
            if not criteria:
                issues.append('Empty scoring criteria')
                quality_score -= 0.4
            else:
                # Check criteria completeness
                points_covered = set()
                for criterion in criteria:
                    if 'points' in criterion:
                        points_covered.add(criterion['points'])
                    if 'criteria' not in criterion or not criterion['criteria'].strip():
                        issues.append('Incomplete criterion description')
                        quality_score -= 0.1

                # Check point range coverage
                max_points = rubric.get('points_possible', 0)
                if max_points not in points_covered:
                    issues.append('Missing maximum points criterion')
                    quality_score -= 0.2

                if 0 not in points_covered:
                    issues.append('Missing zero points criterion')
                    quality_score -= 0.1

        return {
            'quality_score': max(0.0, quality_score),
            'issues': issues,
            'is_complete': len(issues) == 0
        }

    def _calculate_overall_quality_score(self, validation_result: Dict, readability_metrics: Dict,
                                       appropriateness_score: float, difficulty_assessment: Dict,
                                       rubric_quality: Dict) -> float:
        """Calculate overall quality score for a question"""

        weights = {
            'validation': 0.3,
            'readability': 0.2,
            'appropriateness': 0.2,
            'difficulty_alignment': 0.15,
            'rubric_quality': 0.15
        }

        scores = {
            'validation': validation_result.get('quality_score', 0.0),
            'readability': 1.0 if readability_metrics.get('is_appropriate', False) else 0.5,
            'appropriateness': appropriateness_score,
            'difficulty_alignment': difficulty_assessment.get('alignment_score', 0.0),
            'rubric_quality': rubric_quality.get('quality_score', 0.0)
        }

        overall_score = sum(scores[component] * weights[component] for component in weights)
        return overall_score

    def _analyze_question_distribution(self, questions: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of questions across categories"""

        # Subject area distribution
        subject_counts = {}
        difficulty_counts = {}
        type_counts = {}

        for question in questions:
            # Subject areas
            subject = question.get('subject_area', 'unknown')
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

            # Difficulty levels
            difficulty = question.get('difficulty_level', 0)
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

            # Question types
            q_type = question.get('question_type', 'unknown')
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        total_questions = len(questions)

        return {
            'subject_distribution': {k: v/total_questions for k, v in subject_counts.items()},
            'difficulty_distribution': {k: v/total_questions for k, v in difficulty_counts.items()},
            'question_type_distribution': {k: v/total_questions for k, v in type_counts.items()},
            'balance_score': self._calculate_balance_score(subject_counts, difficulty_counts, type_counts)
        }

    def _calculate_balance_score(self, subject_counts: Dict, difficulty_counts: Dict, type_counts: Dict) -> float:
        """Calculate how well-balanced the question distribution is"""

        def entropy(counts):
            total = sum(counts.values())
            if total == 0:
                return 0
            proportions = [count/total for count in counts.values()]
            return -sum(p * np.log2(p) if p > 0 else 0 for p in proportions)

        # Calculate entropy for each distribution (higher = more balanced)
        subject_entropy = entropy(subject_counts)
        difficulty_entropy = entropy(difficulty_counts)
        type_entropy = entropy(type_counts)

        # Normalize to 0-1 scale (assuming max 4-5 categories)
        max_entropy = np.log2(5)  # Assuming max 5 categories

        normalized_entropies = [
            subject_entropy / max_entropy,
            difficulty_entropy / max_entropy,
            type_entropy / max_entropy
        ]

        return np.mean(normalized_entropies)

    def evaluate_scoring_accuracy(self, test_data: Dict, human_scores: Dict[str, Dict]) -> Dict[str, Any]:
        """Evaluate accuracy of automated scoring against human scores"""
        logger.info("Evaluating scoring accuracy...")

        results = {
            'total_comparisons': 0,
            'agreement_metrics': {},
            'detailed_analysis': [],
            'recommendations': []
        }

        automated_scores = []
        human_score_values = []

        for question_id, human_score_data in human_scores.items():
            # Find corresponding question
            question = next((q for q in test_data['questions'] if q['question_id'] == question_id), None)
            if not question:
                continue

            student_response = human_score_data.get('student_response', '')
            human_points = human_score_data.get('points_earned', 0)

            # Get automated score
            automated_result = self.scorer.score_response(question, student_response)
            automated_points = automated_result['points_earned']

            automated_scores.append(automated_points)
            human_score_values.append(human_points)

            # Detailed analysis
            analysis = {
                'question_id': question_id,
                'question_type': question.get('question_type'),
                'human_score': human_points,
                'automated_score': automated_points,
                'difference': abs(human_points - automated_points),
                'agreement': human_points == automated_points,
                'confidence': automated_result.get('confidence', 0.0)
            }

            results['detailed_analysis'].append(analysis)

        results['total_comparisons'] = len(automated_scores)

        if len(automated_scores) > 0:
            # Calculate agreement metrics
            exact_agreement = sum(1 for h, a in zip(human_score_values, automated_scores) if h == a) / len(automated_scores)

            # Calculate correlation
            correlation = np.corrcoef(human_score_values, automated_scores)[0, 1] if len(automated_scores) > 1 else 0

            # Calculate mean absolute error
            mae = np.mean([abs(h - a) for h, a in zip(human_score_values, automated_scores)])

            # Cohen's kappa for categorical agreement
            kappa = cohen_kappa_score(human_score_values, automated_scores)

            results['agreement_metrics'] = {
                'exact_agreement_rate': exact_agreement,
                'correlation': correlation,
                'mean_absolute_error': mae,
                'cohens_kappa': kappa,
                'within_one_point_agreement': sum(1 for h, a in zip(human_score_values, automated_scores)
                                                if abs(h - a) <= 1) / len(automated_scores)
            }

            results['recommendations'] = self._generate_scoring_recommendations(results)

        return results

    def _generate_quality_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Generate recommendations for improving question quality"""
        recommendations = []

        summary = evaluation_results['summary_metrics']

        # Overall quality
        if summary['average_quality_score'] < 0.7:
            recommendations.append("Overall question quality needs improvement. Focus on content appropriateness and clarity.")

        # Readability
        if summary['readability_appropriateness_rate'] < 0.8:
            recommendations.append("Many questions have inappropriate reading levels. Adjust vocabulary and sentence complexity for Year 3.")

        # Distribution balance
        distribution = summary['distribution_analysis']
        if distribution['balance_score'] < 0.6:
            recommendations.append("Question distribution is unbalanced. Ensure even coverage across subjects and difficulty levels.")

        # Low quality questions
        if summary['low_quality_questions'] > summary['total_questions'] * 0.2:
            recommendations.append("Too many low-quality questions. Review and improve questions with quality scores below 0.6.")

        return recommendations

    def _generate_scoring_recommendations(self, scoring_results: Dict) -> List[str]:
        """Generate recommendations for improving scoring accuracy"""
        recommendations = []

        metrics = scoring_results['agreement_metrics']

        if metrics['exact_agreement_rate'] < 0.8:
            recommendations.append("Exact agreement with human scorers is low. Review scoring criteria and algorithms.")

        if metrics['correlation'] < 0.7:
            recommendations.append("Correlation with human scores is weak. Consider improving semantic similarity calculations.")

        if metrics['mean_absolute_error'] > 1.0:
            recommendations.append("Mean absolute error is high. Review scoring rubrics for consistency.")

        if metrics['cohens_kappa'] < 0.6:
            recommendations.append("Inter-rater agreement (kappa) is moderate. Improve scoring reliability.")

        # Analyze by question type
        type_performance = {}
        for analysis in scoring_results['detailed_analysis']:
            q_type = analysis['question_type']
            if q_type not in type_performance:
                type_performance[q_type] = []
            type_performance[q_type].append(analysis['agreement'])

        for q_type, agreements in type_performance.items():
            if np.mean(agreements) < 0.7:
                recommendations.append(f"Scoring accuracy for {q_type} questions is low. Review specific scoring logic.")

        return recommendations

    def generate_evaluation_report(self, evaluation_data: Dict, output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""

        report_sections = []

        # Header
        report_sections.append("# NAPAL System Evaluation Report")
        report_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")

        # Question Quality Analysis
        if 'question_evaluation' in evaluation_data:
            report_sections.append("## Question Quality Analysis")

            quality_data = evaluation_data['question_evaluation']
            summary = quality_data['summary_metrics']

            report_sections.append(f"- **Total Questions Evaluated:** {quality_data['total_questions']}")
            report_sections.append(f"- **Average Quality Score:** {summary['average_quality_score']:.2f}")
            report_sections.append(f"- **High Quality Questions:** {summary['high_quality_questions']}")
            report_sections.append(f"- **Low Quality Questions:** {summary['low_quality_questions']}")
            report_sections.append(f"- **Readability Appropriateness Rate:** {summary['readability_appropriateness_rate']:.1%}")
            report_sections.append("")

            # Recommendations
            if quality_data['recommendations']:
                report_sections.append("### Quality Recommendations")
                for rec in quality_data['recommendations']:
                    report_sections.append(f"- {rec}")
                report_sections.append("")

        # Scoring Accuracy Analysis
        if 'scoring_evaluation' in evaluation_data:
            report_sections.append("## Scoring Accuracy Analysis")

            scoring_data = evaluation_data['scoring_evaluation']
            metrics = scoring_data['agreement_metrics']

            report_sections.append(f"- **Total Comparisons:** {scoring_data['total_comparisons']}")
            report_sections.append(f"- **Exact Agreement Rate:** {metrics['exact_agreement_rate']:.1%}")
            report_sections.append(f"- **Correlation with Human Scores:** {metrics['correlation']:.3f}")
            report_sections.append(f"- **Mean Absolute Error:** {metrics['mean_absolute_error']:.2f}")
            report_sections.append(f"- **Cohen's Kappa:** {metrics['cohens_kappa']:.3f}")
            report_sections.append("")

            # Recommendations
            if scoring_data['recommendations']:
                report_sections.append("### Scoring Recommendations")
                for rec in scoring_data['recommendations']:
                    report_sections.append(f"- {rec}")
                report_sections.append("")

        # Summary and Conclusion
        report_sections.append("## Summary")
        report_sections.append("This evaluation provides insights into the NAPAL system's performance.")
        report_sections.append("Use the recommendations above to improve question generation and scoring accuracy.")

        report_content = "\\n".join(report_sections)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Evaluation report saved to {output_path}")

        return report_content

def main():
    """Main function for running evaluations"""
    evaluator = NAPALEvaluator()

    # Example: Evaluate a generated test
    test_file = Path("generated_tests/sample_test.json")
    if test_file.exists():
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        # Evaluate question quality
        quality_results = evaluator.evaluate_generated_questions(test_data)

        print("Question Quality Evaluation Results:")
        print(f"Average Quality Score: {quality_results['summary_metrics']['average_quality_score']:.2f}")
        print(f"Total Questions: {quality_results['total_questions']}")

        # Generate report
        evaluation_data = {'question_evaluation': quality_results}
        report = evaluator.generate_evaluation_report(evaluation_data)
        print("\\n" + report)

if __name__ == "__main__":
    main()