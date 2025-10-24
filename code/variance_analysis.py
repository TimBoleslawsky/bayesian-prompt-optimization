import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path


class VarianceAnalyzer:
    """Analyzes variance in test results across different prompt types."""
    
    def __init__(self, artefacts_folder="artefacts"):
        self.artefacts_folder = artefacts_folder
        self.test_results = {}
        
    def load_all_test_results(self):
        """Load all test results from the artefacts folder."""
        pattern = f"{self.artefacts_folder}/test_results*.json"
        result_files = glob.glob(pattern)
        
        print(f"Found {len(result_files)} test result files:")
        for file in result_files:
            print(f"  - {file}")
            
        for file_path in result_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract identifier from filename (e.g., test_results_03.json -> "03")
            file_name = Path(file_path).stem
            if file_name == "test_results":
                identifier = "main"
            else:
                identifier = file_name.split("_")[-1]
                
            self.test_results[identifier] = data
            print(f"Loaded {len(data['detailed_results'])} samples from {file_name}")
            
        return self.test_results
    
    def calculate_variance_by_prompt(self):
        """Calculate variance between questions for each prompt type."""
        variance_results = {}
        
        for test_id, data in self.test_results.items():
            print(f"\nAnalyzing variance for test: {test_id}")
            
            # Extract detailed results
            detailed_results = data['detailed_results']
            
            # Initialize variance storage for this test
            variance_results[test_id] = {
                'prompt_variances': {},
                'sample_count': len(detailed_results),
                'metadata': data.get('metadata', {})
            }
            
            # Calculate variance for each prompt type
            for prompt_type in ['original', 'normal', 'bayesian']:
                # Extract scores for this prompt type
                code_quality_scores = []
                faithfulness_scores = []
                combined_scores = []
                
                for result in detailed_results:
                    cq_score = result[f'{prompt_type}_code_quality_score']
                    faith_score = result[f'{prompt_type}_faithfulness_score']
                    combined = cq_score + faith_score
                    
                    code_quality_scores.append(cq_score)
                    faithfulness_scores.append(faith_score)
                    combined_scores.append(combined)
                
                # Calculate variance metrics
                variance_metrics = {
                    'code_quality_variance': float(np.var(code_quality_scores, ddof=1)),
                    'faithfulness_variance': float(np.var(faithfulness_scores, ddof=1)),
                    'combined_score_variance': float(np.var(combined_scores, ddof=1)),
                    'code_quality_std': float(np.std(code_quality_scores, ddof=1)),
                    'faithfulness_std': float(np.std(faithfulness_scores, ddof=1)),
                    'combined_score_std': float(np.std(combined_scores, ddof=1)),
                    'code_quality_mean': float(np.mean(code_quality_scores)),
                    'faithfulness_mean': float(np.mean(faithfulness_scores)),
                    'combined_score_mean': float(np.mean(combined_scores)),
                    'sample_count': len(code_quality_scores)
                }
                
                variance_results[test_id]['prompt_variances'][prompt_type] = variance_metrics
                
                print(f"  {prompt_type.capitalize()} - Combined variance: {variance_metrics['combined_score_variance']:.4f}")
        
        return variance_results
    
    def aggregate_variance_by_prompt(self, variance_results):
        """Aggregate variance metrics across all tests by prompt type."""
        aggregated = {
            'prompt_aggregates': {},
            'cross_test_analysis': {},
            'summary': {}
        }
        
        # Initialize aggregates for each prompt type
        for prompt_type in ['original', 'normal', 'bayesian']:
            all_code_quality_vars = []
            all_faithfulness_vars = []
            all_combined_vars = []
            all_code_quality_means = []
            all_faithfulness_means = []
            all_combined_means = []
            
            # Collect variance metrics across all tests
            for test_id, test_data in variance_results.items():
                if prompt_type in test_data['prompt_variances']:
                    metrics = test_data['prompt_variances'][prompt_type]
                    all_code_quality_vars.append(metrics['code_quality_variance'])
                    all_faithfulness_vars.append(metrics['faithfulness_variance'])
                    all_combined_vars.append(metrics['combined_score_variance'])
                    all_code_quality_means.append(metrics['code_quality_mean'])
                    all_faithfulness_means.append(metrics['faithfulness_mean'])
                    all_combined_means.append(metrics['combined_score_mean'])
            
            # Calculate aggregated statistics
            aggregated['prompt_aggregates'][prompt_type] = {
                'code_quality': {
                    'mean_variance': float(np.mean(all_code_quality_vars)),
                    'variance_of_variances': float(np.var(all_code_quality_vars, ddof=1)) if len(all_code_quality_vars) > 1 else 0.0,
                    'mean_of_means': float(np.mean(all_code_quality_means)),
                    'variance_of_means': float(np.var(all_code_quality_means, ddof=1)) if len(all_code_quality_means) > 1 else 0.0
                },
                'faithfulness': {
                    'mean_variance': float(np.mean(all_faithfulness_vars)),
                    'variance_of_variances': float(np.var(all_faithfulness_vars, ddof=1)) if len(all_faithfulness_vars) > 1 else 0.0,
                    'mean_of_means': float(np.mean(all_faithfulness_means)),
                    'variance_of_means': float(np.var(all_faithfulness_means, ddof=1)) if len(all_faithfulness_means) > 1 else 0.0
                },
                'combined_score': {
                    'mean_variance': float(np.mean(all_combined_vars)),
                    'variance_of_variances': float(np.var(all_combined_vars, ddof=1)) if len(all_combined_vars) > 1 else 0.0,
                    'mean_of_means': float(np.mean(all_combined_means)),
                    'variance_of_means': float(np.var(all_combined_means, ddof=1)) if len(all_combined_means) > 1 else 0.0
                },
                'test_count': len(all_combined_vars)
            }
        
        # Add summary comparison
        aggregated['summary'] = {
            'most_consistent_prompt': self._find_most_consistent_prompt(aggregated['prompt_aggregates']),
            'highest_variance_prompt': self._find_highest_variance_prompt(aggregated['prompt_aggregates']),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_tests_analyzed': len(variance_results)
        }
        
        # Add the raw variance results for reference
        aggregated['detailed_variance_by_test'] = variance_results
        
        return aggregated
    
    def _find_most_consistent_prompt(self, prompt_aggregates):
        """Find the prompt with lowest combined variance."""
        min_variance = float('inf')
        most_consistent = None
        
        for prompt_type, metrics in prompt_aggregates.items():
            combined_variance = metrics['combined_score']['mean_variance']
            if combined_variance < min_variance:
                min_variance = combined_variance
                most_consistent = prompt_type
                
        return {
            'prompt_type': most_consistent,
            'mean_variance': min_variance
        }
    
    def _find_highest_variance_prompt(self, prompt_aggregates):
        """Find the prompt with highest combined variance."""
        max_variance = 0.0
        highest_variance = None
        
        for prompt_type, metrics in prompt_aggregates.items():
            combined_variance = metrics['combined_score']['mean_variance']
            if combined_variance > max_variance:
                max_variance = combined_variance
                highest_variance = prompt_type
                
        return {
            'prompt_type': highest_variance,
            'mean_variance': max_variance
        }
    
    def save_variance_analysis(self, variance_analysis, filename="variance_analysis.json"):
        """Save variance analysis results to JSON file."""
        output_path = f"{self.artefacts_folder}/{filename}"
        
        with open(output_path, 'w') as f:
            json.dump(variance_analysis, f, indent=2)
            
        print(f"\nVariance analysis saved to: {output_path}")
        return output_path
    
    def print_summary(self, variance_analysis):
        """Print a summary of the variance analysis."""
        print("\n" + "="*80)
        print("VARIANCE ANALYSIS SUMMARY")
        print("="*80)
        
        summary = variance_analysis['summary']
        print(f"Total tests analyzed: {summary['total_tests_analyzed']}")
        print(f"Analysis timestamp: {summary['analysis_timestamp']}")
        
        print(f"\nMost consistent prompt: {summary['most_consistent_prompt']['prompt_type']}")
        print(f"  Mean combined variance: {summary['most_consistent_prompt']['mean_variance']:.4f}")
        
        print(f"\nHighest variance prompt: {summary['highest_variance_prompt']['prompt_type']}")
        print(f"  Mean combined variance: {summary['highest_variance_prompt']['mean_variance']:.4f}")
        
        print("\nDetailed variance by prompt type:")
        print("-" * 60)
        
        for prompt_type, metrics in variance_analysis['prompt_aggregates'].items():
            print(f"\n{prompt_type.upper()}:")
            print(f"  Code Quality - Mean variance: {metrics['code_quality']['mean_variance']:.4f}")
            print(f"  Faithfulness - Mean variance: {metrics['faithfulness']['mean_variance']:.4f}")
            print(f"  Combined     - Mean variance: {metrics['combined_score']['mean_variance']:.4f}")
            print(f"  Tests analyzed: {metrics['test_count']}")


def main():
    """Main function to run variance analysis."""
    analyzer = VarianceAnalyzer()
    
    print("Loading test results...")
    test_results = analyzer.load_all_test_results()
    
    if not test_results:
        print("No test results found in artefacts folder!")
        return
    
    print("\nCalculating variance by prompt...")
    variance_results = analyzer.calculate_variance_by_prompt()
    
    print("\nAggregating variance by prompt...")
    variance_analysis = analyzer.aggregate_variance_by_prompt(variance_results)
    
    # Print summary
    analyzer.print_summary(variance_analysis)
    
    # Save results
    analyzer.save_variance_analysis(variance_analysis)
    
    print("\nVariance analysis complete!")


if __name__ == "__main__":
    main()
