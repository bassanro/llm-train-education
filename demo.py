"""
Example script showing how to use the NAPAL LLM system end-to-end
"""

import json
from pathlib import Path
import sys

# Add project root to path (so `src` is a package and internal relative imports work)
sys.path.insert(0, str(Path(__file__).parent))

# Guard imports and give a helpful message if dependencies are missing
try:
    from src.generation.test_generator import NAPALTestGenerator
    from src.scoring.scorer import NAPALScorer
    from src.evaluation.evaluator import NAPALEvaluator
    from src.workflow.annual_manager import AnnualWorkflowManager
except ModuleNotFoundError as e:
    missing = e.name if hasattr(e, 'name') else str(e)
    print(f"ERROR: Missing Python dependency: {missing}\n")
    print("Quick fixes:")
    print("  1) If you use a virtual environment, activate it, e.g.:\n     source napal_env/bin/activate")
    print("  2) Install project dependencies into the active environment:\n     python3 -m pip install --upgrade pip && pip install -r requirements.txt")
    print("  3) If you only need the missing package, install it directly, e.g.:\n     python3 -m pip install pyyaml  # or the package name shown above")
    print("After installing, re-run: python3 demo.py\n")
    sys.exit(1)

def demo_test_generation():
    """Demonstrate test generation"""
    print("=== NAPAL Test Generation Demo ===\\n")

    try:
        # Initialize generator
        print("1. Initializing test generator...")
        generator = NAPALTestGenerator()

        # Generate a small test
        print("2. Generating test for year 2025...")
        test_data = generator.generate_test(year=2025, num_questions=5)

        print(f"✓ Generated test with {len(test_data['questions'])} questions")
        print(f"  Test ID: {test_data['test_metadata']['test_id']}")
        print(f"  Estimated duration: {test_data['test_metadata']['estimated_duration_minutes']} minutes")

        # Save test
        output_path = Path("generated_tests") / "demo_test.json"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"✓ Test saved to {output_path}")

        return test_data, str(output_path)

    except Exception as e:
        print(f"✗ Error in test generation: {e}")
        return None, None

def demo_test_evaluation(test_data):
    """Demonstrate test evaluation"""
    print("\\n=== Test Quality Evaluation Demo ===\\n")

    try:
        # Initialize evaluator
        print("3. Evaluating test quality...")
        evaluator = NAPALEvaluator()

        # Evaluate the test
        evaluation_results = evaluator.evaluate_generated_questions(test_data)

        print(f"✓ Evaluation completed")
        print(f"  Overall quality score: {evaluation_results['overall_quality_score']:.3f}")
        print(f"  Average readability: {evaluation_results['summary_metrics']['average_readability']:.1f}")
        print(f"  High quality questions: {evaluation_results['summary_metrics']['high_quality_questions']}")

        # Show question-level details
        print("\\n  Question details:")
        for i, q_eval in enumerate(evaluation_results['question_evaluations'][:3]):  # Show first 3
            print(f"    Q{i+1}: Quality={q_eval['quality_score']:.2f}, "
                  f"Readability={q_eval['readability_metrics']['readability_level']:.1f}")

        return evaluation_results

    except Exception as e:
        print(f"✗ Error in test evaluation: {e}")
        return None

def demo_scoring():
    """Demonstrate scoring system"""
    print("\\n=== Automated Scoring Demo ===\\n")

    try:
        # Create sample student responses
        sample_responses = {
            "Q001": "B",  # Multiple choice
            "Q002": "happy",  # Short answer
            "Q003": "The cat is very cute and fluffy. It likes to play with toys."  # Extended response
        }

        print("4. Testing automated scoring...")

        # Load the demo test
        test_file = Path("generated_tests") / "demo_test.json"
        if not test_file.exists():
            print("✗ Demo test file not found. Run test generation first.")
            return None

        with open(test_file, 'r') as f:
            test_data = json.load(f)

        # Initialize scorer
        scorer = NAPALScorer()

        # Score individual questions (first few for demo)
        print("  Scoring individual questions:")
        for i, question in enumerate(test_data['questions'][:3]):
            q_id = question['question_id']
            if q_id in sample_responses:
                response = sample_responses[q_id]
                result = scorer.score_response(question, response)

                print(f"    {q_id}: {result['points_earned']}/{result['points_possible']} points "
                      f"(Confidence: {result['confidence']:.2f})")

        print("✓ Scoring demonstration completed")

        return True

    except Exception as e:
        print(f"✗ Error in scoring demo: {e}")
        return None

def demo_annual_workflow():
    """Demonstrate annual workflow management"""
    print("\\n=== Annual Workflow Demo ===\\n")

    try:
        print("5. Demonstrating annual workflow...")

        # Initialize workflow manager
        manager = AnnualWorkflowManager()

        # Check current status
        years = manager.list_all_years()
        print(f"  Currently generated years: {len(years)}")

        # Get next recommended year
        next_year = manager.get_next_recommended_year()
        print(f"  Next recommended year: {next_year}")

        # Check if we can generate a small demo (1 variant, 3 questions)
        print(f"  Demo: Would generate tests for {next_year}")
        print(f"  (Use 'python main.py workflow --generate {next_year}' to actually generate)")

        return True

    except Exception as e:
        print(f"✗ Error in workflow demo: {e}")
        return None

def main():
    """Run the complete demo"""
    print("NAPAL LLM Test Generation System - Demo\\n")
    print("This demo shows the key features of the system:\\n")

    # Run demonstrations
    test_data, test_path = demo_test_generation()

    if test_data:
        demo_test_evaluation(test_data)
        demo_scoring()

    demo_annual_workflow()

    print("\\n=== Demo Completed ===")
    print("\\nNext steps:")
    print("1. Add your own NAPAL training data to data/raw/")
    print("2. Run: python main.py prepare-data")
    print("3. Run: python main.py fine-tune")
    print("4. Run: python main.py generate --year 2025")
    print("\\nFor more options, run: python main.py --help")

if __name__ == "__main__":
    main()