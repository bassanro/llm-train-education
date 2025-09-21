"""
Tests for the NAPAL LLM system components
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.utils import TextProcessor, QuestionValidator
from scoring.scorer import NAPALScorer

class TestTextProcessor:
    """Test TextProcessor functionality"""

    def setup_method(self):
        self.processor = TextProcessor()

    def test_clean_text(self):
        text = "  Hello,   world!  "
        cleaned = self.processor.clean_text(text)
        assert cleaned == "Hello, world!"

    def test_readability_calculation(self):
        # Simple text appropriate for Year 3
        text = "The cat sat on the mat. It was happy."
        readability = self.processor.calculate_readability_level(text)
        assert 1.0 <= readability <= 5.0

    def test_age_appropriateness_check(self):
        # Age-appropriate text
        appropriate_text = "Emma loves to read books about animals."
        result = self.processor.check_age_appropriateness(appropriate_text)
        assert isinstance(result, dict)
        assert 'readability_level' in result
        assert 'is_appropriate' in result

class TestQuestionValidator:
    """Test QuestionValidator functionality"""

    def setup_method(self):
        self.validator = QuestionValidator()

    def test_valid_question(self):
        question = {
            "question_text": "What color is the sky?",
            "correct_answer": {"answer": "blue"},
            "scoring_rubric": {"points_possible": 1}
        }
        result = self.validator.validate_question(question)
        assert result["is_valid"] is True

    def test_invalid_question_missing_fields(self):
        question = {
            "question_text": "What color is the sky?"
            # Missing required fields
        }
        result = self.validator.validate_question(question)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0

class TestNAPALScorer:
    """Test NAPALScorer functionality"""

    def setup_method(self):
        # Create temporary config
        self.config_data = {
            'scoring': {
                'similarity_threshold': 0.8,
                'use_semantic_matching': True
            }
        }

        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        import yaml
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()

        self.scorer = NAPALScorer(self.temp_config.name)

    def teardown_method(self):
        # Clean up temp file
        Path(self.temp_config.name).unlink()

    def test_multiple_choice_scoring(self):
        question = {
            "question_type": "multiple_choice",
            "correct_answer": {"answer": "B"},
            "scoring_rubric": {"points_possible": 1}
        }

        # Correct answer
        result = self.scorer.score_response(question, "B")
        assert result["points_earned"] == 1
        assert result["is_correct"] is True

        # Incorrect answer
        result = self.scorer.score_response(question, "A")
        assert result["points_earned"] == 0
        assert result["is_correct"] is False

    def test_short_answer_scoring(self):
        question = {
            "question_type": "short_answer",
            "correct_answer": {
                "answer": "blue",
                "acceptable_variations": ["azure", "sky-blue"]
            },
            "scoring_rubric": {"points_possible": 2}
        }

        # Exact match
        result = self.scorer.score_response(question, "blue")
        assert result["points_earned"] == 2

        # Acceptable variation
        result = self.scorer.score_response(question, "azure")
        assert result["points_earned"] == 2

        # Partial match
        result = self.scorer.score_response(question, "kind of blue")
        assert result["points_earned"] >= 1

class TestSampleData:
    """Test with sample NAPAL data"""

    def test_sample_test_format(self):
        """Test that our sample test follows the expected format"""
        sample_file = Path(__file__).parent.parent / "data" / "raw" / "sample_napal_test.json"

        assert sample_file.exists(), "Sample test file should exist"

        with open(sample_file, 'r') as f:
            test_data = json.load(f)

        # Check required top-level structure
        assert "test_metadata" in test_data
        assert "questions" in test_data

        # Check metadata structure
        metadata = test_data["test_metadata"]
        required_metadata_fields = ["test_id", "year", "grade_level", "subject_areas", "total_questions"]
        for field in required_metadata_fields:
            assert field in metadata, f"Required metadata field '{field}' missing"

        # Check questions structure
        questions = test_data["questions"]
        assert len(questions) > 0, "Should have at least one question"

        for question in questions:
            required_question_fields = ["question_id", "question_type", "subject_area", "question_text", "correct_answer", "scoring_rubric"]
            for field in required_question_fields:
                assert field in question, f"Required question field '{field}' missing"

# Integration tests
class TestSystemIntegration:
    """Test system components working together"""

    def test_end_to_end_workflow(self):
        """Test a simplified end-to-end workflow"""
        # This would test the complete pipeline:
        # 1. Load sample data
        # 2. Generate questions (mocked for testing)
        # 3. Score responses
        # 4. Evaluate quality

        # For now, just test that components can be imported and initialized
        from generation.test_generator import NAPALTestGenerator
        from evaluation.evaluator import NAPALEvaluator
        from workflow.annual_manager import AnnualWorkflowManager

        # Test that we can create instances (may fail if dependencies missing)
        try:
            # These will likely fail without trained models, but should import
            assert NAPALTestGenerator is not None
            assert NAPALEvaluator is not None
            assert AnnualWorkflowManager is not None

        except Exception as e:
            # Expected if dependencies not installed or models not trained
            pytest.skip(f"Integration test skipped due to missing dependencies: {e}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])