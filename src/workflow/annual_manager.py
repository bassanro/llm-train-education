"""
Annual generation workflow for NAPAL tests.
Manages yearly test creation, tracking, and archival.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import uuid
import shutil

from ..generation.test_generator import NAPALTestGenerator
from ..evaluation.evaluator import NAPALEvaluator
from ..scoring.scorer import NAPALScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnnualWorkflowManager:
    """Manages annual NAPAL test generation workflow"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.base_dir = Path(self.config['paths']['output_dir'])
        self.archive_dir = self.base_dir / "archive"
        self.tracking_file = self.base_dir / "generation_tracking.json"

        # Initialize components
        self.generator = NAPALTestGenerator(config_path)
        self.evaluator = NAPALEvaluator(config_path)
        self.scorer = NAPALScorer(config_path)

        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Load or create tracking data
        self.tracking_data = self._load_tracking_data()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_tracking_data(self) -> Dict:
        """Load generation tracking data"""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "created_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "annual_generations": {},
                "statistics": {
                    "total_tests_generated": 0,
                    "total_questions_generated": 0,
                    "average_quality_score": 0.0
                }
            }

    def _save_tracking_data(self):
        """Save tracking data to file"""
        self.tracking_data["last_updated"] = datetime.now().isoformat()
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)

    def generate_annual_test_suite(self, year: int,
                                  num_test_variants: int = 3,
                                  questions_per_test: int = 20) -> Dict[str, Any]:
        """Generate multiple test variants for a given year"""

        logger.info(f"Starting annual test generation for year {year}")

        # Check if year already exists
        if str(year) in self.tracking_data["annual_generations"]:
            logger.warning(f"Tests for year {year} already exist. Use regenerate_year() to overwrite.")
            return self.tracking_data["annual_generations"][str(year)]

        # Create year directory
        year_dir = self.base_dir / str(year)
        year_dir.mkdir(exist_ok=True)

        # Generate multiple test variants
        test_variants = []
        generation_results = {
            "year": year,
            "generation_date": datetime.now().isoformat(),
            "num_variants": num_test_variants,
            "questions_per_test": questions_per_test,
            "test_variants": [],
            "quality_metrics": {},
            "status": "in_progress"
        }

        for variant_num in range(1, num_test_variants + 1):
            logger.info(f"Generating test variant {variant_num}/{num_test_variants}")

            # Generate test with slight variations
            custom_distribution = self._create_variant_distribution(variant_num)

            test_data = self.generator.generate_test(
                year=year,
                num_questions=questions_per_test,
                custom_distribution=custom_distribution
            )

            # Update test metadata
            test_data['test_metadata']['variant'] = variant_num
            test_data['test_metadata']['test_id'] = f"NAPAL_Y3_{year}_V{variant_num:02d}_{datetime.now().strftime('%m%d')}"

            # Save test
            test_filename = f"test_variant_{variant_num:02d}.json"
            test_filepath = year_dir / test_filename

            with open(test_filepath, 'w') as f:
                json.dump(test_data, f, indent=2)

            # Evaluate test quality
            quality_eval = self.evaluator.evaluate_generated_questions(test_data)

            variant_info = {
                "variant_number": variant_num,
                "test_id": test_data['test_metadata']['test_id'],
                "filename": test_filename,
                "filepath": str(test_filepath),
                "num_questions": len(test_data['questions']),
                "quality_score": quality_eval['overall_quality_score'],
                "subject_distribution": self._calculate_subject_distribution(test_data['questions']),
                "difficulty_distribution": self._calculate_difficulty_distribution(test_data['questions']),
                "generated_at": test_data['test_metadata']['created_date']
            }

            generation_results["test_variants"].append(variant_info)
            test_variants.append(test_data)

            logger.info(f"Variant {variant_num} completed with quality score: {quality_eval['overall_quality_score']:.2f}")

        # Calculate overall quality metrics
        quality_scores = [variant['quality_score'] for variant in generation_results["test_variants"]]
        generation_results["quality_metrics"] = {
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "quality_variance": self._calculate_variance(quality_scores)
        }

        # Generate evaluation report
        self._generate_annual_report(year, generation_results, test_variants)

        # Update tracking
        generation_results["status"] = "completed"
        self.tracking_data["annual_generations"][str(year)] = generation_results

        # Update statistics
        self._update_statistics(generation_results)

        # Save tracking data
        self._save_tracking_data()

        logger.info(f"Annual test generation for {year} completed successfully")
        return generation_results

    def _create_variant_distribution(self, variant_num: int) -> Dict[str, Dict]:
        """Create slightly different distributions for test variants"""
        base_config = self.config['generation']

        # Create variations while maintaining overall balance
        variations = {
            1: {  # Variant 1: Balanced
                'subject_distribution': base_config['subject_distribution'],
                'difficulty_distribution': base_config['difficulty_distribution'],
                'question_type_distribution': base_config['question_type_distribution']
            },
            2: {  # Variant 2: More reading comprehension
                'subject_distribution': {
                    'reading_comprehension': 0.5,
                    'vocabulary': 0.25,
                    'writing': 0.15,
                    'language_conventions': 0.1
                },
                'difficulty_distribution': base_config['difficulty_distribution'],
                'question_type_distribution': base_config['question_type_distribution']
            },
            3: {  # Variant 3: More writing focus
                'subject_distribution': {
                    'reading_comprehension': 0.3,
                    'vocabulary': 0.25,
                    'writing': 0.35,
                    'language_conventions': 0.1
                },
                'difficulty_distribution': base_config['difficulty_distribution'],
                'question_type_distribution': {
                    'multiple_choice': 0.4,
                    'short_answer': 0.35,
                    'extended_response': 0.25
                }
            }
        }

        return variations.get(variant_num, variations[1])

    def _calculate_subject_distribution(self, questions: List[Dict]) -> Dict[str, float]:
        """Calculate actual subject distribution"""
        subject_counts = {}
        for question in questions:
            subject = question.get('subject_area', 'unknown')
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        total = len(questions)
        return {subject: count/total for subject, count in subject_counts.items()}

    def _calculate_difficulty_distribution(self, questions: List[Dict]) -> Dict[str, float]:
        """Calculate actual difficulty distribution"""
        difficulty_counts = {}
        for question in questions:
            difficulty = question.get('difficulty_level', 0)
            key = f"level_{difficulty}"
            difficulty_counts[key] = difficulty_counts.get(key, 0) + 1

        total = len(questions)
        return {level: count/total for level, count in difficulty_counts.items()}

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _generate_annual_report(self, year: int, generation_results: Dict, test_variants: List[Dict]):
        """Generate comprehensive annual report"""
        year_dir = self.base_dir / str(year)
        report_path = year_dir / f"annual_report_{year}.md"

        report_sections = []

        # Header
        report_sections.append(f"# NAPAL Annual Test Generation Report - {year}")
        report_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")

        # Summary
        report_sections.append("## Generation Summary")
        report_sections.append(f"- **Year:** {year}")
        report_sections.append(f"- **Number of Test Variants:** {generation_results['num_variants']}")
        report_sections.append(f"- **Questions per Test:** {generation_results['questions_per_test']}")
        report_sections.append(f"- **Average Quality Score:** {generation_results['quality_metrics']['average_quality']:.3f}")
        report_sections.append(f"- **Quality Range:** {generation_results['quality_metrics']['min_quality']:.3f} - {generation_results['quality_metrics']['max_quality']:.3f}")
        report_sections.append("")

        # Variant Details
        report_sections.append("## Test Variant Details")
        for variant in generation_results['test_variants']:
            report_sections.append(f"### Variant {variant['variant_number']}")
            report_sections.append(f"- **Test ID:** {variant['test_id']}")
            report_sections.append(f"- **Quality Score:** {variant['quality_score']:.3f}")
            report_sections.append(f"- **Subject Distribution:**")
            for subject, ratio in variant['subject_distribution'].items():
                report_sections.append(f"  - {subject}: {ratio:.1%}")
            report_sections.append("")

        # Quality Analysis
        report_sections.append("## Quality Analysis")

        # Evaluate first variant as representative
        if test_variants:
            quality_eval = self.evaluator.evaluate_generated_questions(test_variants[0])

            if quality_eval['recommendations']:
                report_sections.append("### Recommendations for Improvement")
                for rec in quality_eval['recommendations']:
                    report_sections.append(f"- {rec}")
                report_sections.append("")

        # Usage Instructions
        report_sections.append("## Usage Instructions")
        report_sections.append("1. Select the appropriate test variant for your assessment needs")
        report_sections.append("2. Review questions for final approval before administration")
        report_sections.append("3. Use the automated scoring system for consistent evaluation")
        report_sections.append("4. Archive used tests and select new variants for subsequent assessments")

        # Write report
        report_content = "\\n".join(report_sections)
        with open(report_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Annual report saved to {report_path}")

    def _update_statistics(self, generation_results: Dict):
        """Update overall statistics"""
        stats = self.tracking_data["statistics"]

        stats["total_tests_generated"] += generation_results["num_variants"]
        stats["total_questions_generated"] += generation_results["num_variants"] * generation_results["questions_per_test"]

        # Update average quality score
        all_scores = []
        for year_data in self.tracking_data["annual_generations"].values():
            if year_data.get("status") == "completed":
                year_scores = [variant['quality_score'] for variant in year_data['test_variants']]
                all_scores.extend(year_scores)

        if all_scores:
            stats["average_quality_score"] = sum(all_scores) / len(all_scores)

    def regenerate_year(self, year: int, force: bool = False) -> Dict[str, Any]:
        """Regenerate tests for a specific year"""

        if str(year) in self.tracking_data["annual_generations"] and not force:
            raise ValueError(f"Year {year} already exists. Use force=True to overwrite.")

        # Archive existing if present
        if str(year) in self.tracking_data["annual_generations"]:
            self._archive_year(year)

        # Remove from tracking
        if str(year) in self.tracking_data["annual_generations"]:
            del self.tracking_data["annual_generations"][str(year)]

        # Regenerate
        return self.generate_annual_test_suite(year)

    def _archive_year(self, year: int):
        """Archive tests for a specific year"""
        year_dir = self.base_dir / str(year)
        if year_dir.exists():
            archive_year_dir = self.archive_dir / f"{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(year_dir), str(archive_year_dir))
            logger.info(f"Archived {year} tests to {archive_year_dir}")

    def get_year_status(self, year: int) -> Dict[str, Any]:
        """Get status and information for a specific year"""
        year_str = str(year)

        if year_str not in self.tracking_data["annual_generations"]:
            return {"status": "not_generated", "year": year}

        return self.tracking_data["annual_generations"][year_str]

    def list_all_years(self) -> List[Dict[str, Any]]:
        """List all generated years with summary information"""
        years_info = []

        for year_str, year_data in self.tracking_data["annual_generations"].items():
            summary = {
                "year": int(year_str),
                "status": year_data.get("status", "unknown"),
                "num_variants": year_data.get("num_variants", 0),
                "average_quality": year_data.get("quality_metrics", {}).get("average_quality", 0.0),
                "generation_date": year_data.get("generation_date", "unknown")
            }
            years_info.append(summary)

        return sorted(years_info, key=lambda x: x["year"], reverse=True)

    def get_next_recommended_year(self) -> int:
        """Get the next recommended year for generation"""
        current_year = datetime.now().year

        # Check if current year exists
        if str(current_year) not in self.tracking_data["annual_generations"]:
            return current_year

        # Return next year
        return current_year + 1

    def schedule_automatic_generation(self, enable: bool = True,
                                    generation_month: int = 1,
                                    generation_day: int = 15) -> Dict[str, Any]:
        """Configure automatic annual generation"""

        schedule_config = {
            "enabled": enable,
            "generation_month": generation_month,
            "generation_day": generation_day,
            "last_auto_generation": None,
            "next_scheduled": None
        }

        if enable:
            # Calculate next scheduled generation
            current_date = datetime.now()
            current_year = current_date.year

            scheduled_date = datetime(current_year, generation_month, generation_day)

            # If this year's scheduled date has passed, schedule for next year
            if scheduled_date <= current_date:
                scheduled_date = datetime(current_year + 1, generation_month, generation_day)

            schedule_config["next_scheduled"] = scheduled_date.isoformat()

        # Save to tracking data
        self.tracking_data["automatic_schedule"] = schedule_config
        self._save_tracking_data()

        return schedule_config

    def check_and_run_scheduled_generation(self) -> Optional[Dict[str, Any]]:
        """Check if scheduled generation should run and execute if needed"""

        schedule = self.tracking_data.get("automatic_schedule", {})

        if not schedule.get("enabled", False):
            return None

        next_scheduled = schedule.get("next_scheduled")
        if not next_scheduled:
            return None

        scheduled_date = datetime.fromisoformat(next_scheduled)
        current_date = datetime.now()

        if current_date >= scheduled_date:
            # Run scheduled generation
            year = scheduled_date.year

            logger.info(f"Running scheduled generation for year {year}")

            try:
                result = self.generate_annual_test_suite(year)

                # Update schedule
                schedule["last_auto_generation"] = current_date.isoformat()
                next_year_date = datetime(year + 1, schedule["generation_month"], schedule["generation_day"])
                schedule["next_scheduled"] = next_year_date.isoformat()

                self.tracking_data["automatic_schedule"] = schedule
                self._save_tracking_data()

                return result

            except Exception as e:
                logger.error(f"Scheduled generation failed: {e}")
                return None

        return None

    def export_tracking_summary(self, output_path: Optional[str] = None) -> str:
        """Export summary of all tracking data"""

        if output_path is None:
            output_path = self.base_dir / "tracking_summary.json"

        summary = {
            "system_info": {
                "created_date": self.tracking_data["created_date"],
                "last_updated": self.tracking_data["last_updated"],
                "total_years_generated": len(self.tracking_data["annual_generations"])
            },
            "statistics": self.tracking_data["statistics"],
            "years_summary": self.list_all_years(),
            "automatic_schedule": self.tracking_data.get("automatic_schedule", {}),
            "export_date": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Tracking summary exported to {output_path}")
        return str(output_path)

def main():
    """Main function for workflow management"""
    import argparse

    parser = argparse.ArgumentParser(description="NAPAL Annual Workflow Manager")
    parser.add_argument("--generate", type=int, help="Generate tests for specified year")
    parser.add_argument("--regenerate", type=int, help="Regenerate tests for specified year")
    parser.add_argument("--list", action="store_true", help="List all generated years")
    parser.add_argument("--status", type=int, help="Get status for specified year")
    parser.add_argument("--check-schedule", action="store_true", help="Check and run scheduled generation")
    parser.add_argument("--variants", type=int, default=3, help="Number of test variants to generate")
    parser.add_argument("--questions", type=int, default=20, help="Number of questions per test")

    args = parser.parse_args()

    manager = AnnualWorkflowManager()

    if args.generate:
        result = manager.generate_annual_test_suite(
            year=args.generate,
            num_test_variants=args.variants,
            questions_per_test=args.questions
        )
        print(f"Generated {result['num_variants']} test variants for year {args.generate}")
        print(f"Average quality score: {result['quality_metrics']['average_quality']:.3f}")

    elif args.regenerate:
        result = manager.regenerate_year(args.regenerate, force=True)
        print(f"Regenerated tests for year {args.regenerate}")

    elif args.list:
        years = manager.list_all_years()
        print("Generated Years:")
        for year_info in years:
            print(f"  {year_info['year']}: {year_info['status']} (Quality: {year_info['average_quality']:.3f})")

    elif args.status:
        status = manager.get_year_status(args.status)
        print(f"Year {args.status} status: {status}")

    elif args.check_schedule:
        result = manager.check_and_run_scheduled_generation()
        if result:
            print(f"Scheduled generation completed for year {result['year']}")
        else:
            print("No scheduled generation needed")

    else:
        # Show next recommended year
        next_year = manager.get_next_recommended_year()
        print(f"Next recommended year for generation: {next_year}")

if __name__ == "__main__":
    main()