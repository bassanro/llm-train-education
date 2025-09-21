"""
Main entry point for NAPAL LLM System
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path so `src` is importable as a package
sys.path.insert(0, str(Path(__file__).parent))

from src.training.prepare_data import main as prepare_data_main
from src.training.fine_tune import main as fine_tune_main
from src.generation.test_generator import main as generate_test_main
from src.scoring.scorer import main as score_test_main
from src.evaluation.evaluator import main as evaluate_main
from src.workflow.annual_manager import main as workflow_main

def main():
    """Main CLI interface for NAPAL LLM System"""

    parser = argparse.ArgumentParser(
        description="NAPAL LLM Test Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare training data
  python main.py prepare-data

  # Fine-tune the model
  python main.py fine-tune

  # Generate a test for 2025
  python main.py generate --year 2025 --num-questions 20

  # Score student responses
  python main.py score --test-file test.json --responses responses.json

  # Evaluate generated questions
  python main.py evaluate --test-file test.json

  # Manage annual workflow
  python main.py workflow --generate 2025
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Data preparation command
    prep_parser = subparsers.add_parser('prepare-data', help='Prepare training data')

    # Fine-tuning command
    tune_parser = subparsers.add_parser('fine-tune', help='Fine-tune the model')
    tune_parser.add_argument('--config', default='configs/config.yaml', help='Config file path')

    # Test generation command
    gen_parser = subparsers.add_parser('generate', help='Generate test questions')
    gen_parser.add_argument('--year', type=int, required=True, help='Year for the test')
    gen_parser.add_argument('--num-questions', type=int, help='Number of questions')
    gen_parser.add_argument('--output-dir', help='Output directory')
    gen_parser.add_argument('--config', default='configs/config.yaml', help='Config file path')

    # Scoring command
    score_parser = subparsers.add_parser('score', help='Score student responses')
    score_parser.add_argument('--test-file', required=True, help='Test JSON file')
    score_parser.add_argument('--responses', required=True, help='Student responses JSON file')
    score_parser.add_argument('--output', help='Output file for scores')

    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate generated questions')
    eval_parser.add_argument('--test-file', required=True, help='Test JSON file to evaluate')
    eval_parser.add_argument('--output', help='Output file for evaluation report')

    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Manage annual workflow')
    workflow_parser.add_argument('--generate', type=int, help='Generate tests for year')
    workflow_parser.add_argument('--list', action='store_true', help='List all years')
    workflow_parser.add_argument('--status', type=int, help='Get status for year')
    workflow_parser.add_argument('--variants', type=int, default=3, help='Number of variants')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'prepare-data':
            prepare_data_main()

        elif args.command == 'fine-tune':
            # Set sys.argv for the fine-tune main function
            original_argv = sys.argv
            sys.argv = ['fine_tune.py']
            if hasattr(args, 'config'):
                sys.argv.extend(['--config', args.config])
            fine_tune_main()
            sys.argv = original_argv

        elif args.command == 'generate':
            original_argv = sys.argv
            sys.argv = ['test_generator.py', '--year', str(args.year)]
            if args.num_questions:
                sys.argv.extend(['--num_questions', str(args.num_questions)])
            if args.output_dir:
                sys.argv.extend(['--output_dir', args.output_dir])
            if args.config:
                sys.argv.extend(['--config', args.config])
            generate_test_main()
            sys.argv = original_argv

        elif args.command == 'score':
            score_student_responses(args.test_file, args.responses, args.output)

        elif args.command == 'evaluate':
            evaluate_test_file(args.test_file, args.output)

        elif args.command == 'workflow':
            original_argv = sys.argv
            sys.argv = ['annual_manager.py']
            if args.generate:
                sys.argv.extend(['--generate', str(args.generate)])
            if args.list:
                sys.argv.append('--list')
            if args.status:
                sys.argv.extend(['--status', str(args.status)])
            if args.variants:
                sys.argv.extend(['--variants', str(args.variants)])
            workflow_main()
            sys.argv = original_argv

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def score_student_responses(test_file: str, responses_file: str, output_file: str = None):
    """Score student responses using the scoring system"""
    import json
    from src.scoring.scorer import NAPALScorer

    # Load test data
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Load student responses
    with open(responses_file, 'r') as f:
        student_responses = json.load(f)

    # Initialize scorer and score test
    scorer = NAPALScorer()
    results = scorer.score_test(test_data, student_responses)

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Scoring results saved to {output_file}")
    else:
        print(f"Overall Score: {results['percentage_score']:.1f}%")
        print(f"Points: {results['total_points_earned']}/{results['total_points_possible']}")

def evaluate_test_file(test_file: str, output_file: str = None):
    """Evaluate a test file for quality"""
    import json
    from src.evaluation.evaluator import NAPALEvaluator

    # Load test data
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Initialize evaluator and evaluate
    evaluator = NAPALEvaluator()
    results = evaluator.evaluate_generated_questions(test_data)

    # Generate report
    evaluation_data = {'question_evaluation': results}
    report = evaluator.generate_evaluation_report(evaluation_data, output_file)

    if not output_file:
        print(report)
    else:
        print(f"Evaluation report saved to {output_file}")

if __name__ == "__main__":
    main()