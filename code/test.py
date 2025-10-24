import json
import pandas as pd
import dspy
import argparse
from settings import settings
from code.optimization import (
    CodeQualityPredictor,
    FaithfulnessPredictor
)


class PromptLoader:
    """Helper class to load and manage different prompt versions."""
    
    @staticmethod
    def load_original_modules():
        """Load original (unoptimized) modules."""
        return {
            'code_quality': CodeQualityPredictor(),
            'faithfulness': FaithfulnessPredictor()
        }
    
    @staticmethod
    def load_optimized_module(prompt_path, module_class):
        """Load an optimized module from saved JSON."""
        try:
            module = module_class()
            module.load(prompt_path)
            return module
        except Exception as e:
            print(f"Warning: Could not load {prompt_path}: {e}")
            return None
    
    @staticmethod
    def load_all_prompts():
        """Load all prompt variants for testing."""
        prompts = {
            'code_quality': {},
            'faithfulness': {}
        }
        
        # Load original (unoptimized) prompts
        original = PromptLoader.load_original_modules()
        prompts['code_quality']['original'] = original['code_quality']
        prompts['faithfulness']['original'] = original['faithfulness']
        
        # Load Bayesian optimized prompts
        bayesian_cq = PromptLoader.load_optimized_module(
            "prompts/code_quality_bayesian.json",
            CodeQualityPredictor
        )
        bayesian_faith = PromptLoader.load_optimized_module(
            "prompts/faithfulness_bayesian.json",
            FaithfulnessPredictor
        )
        
        if bayesian_cq:
            prompts['code_quality']['bayesian'] = bayesian_cq
        if bayesian_faith:
            prompts['faithfulness']['bayesian'] = bayesian_faith
        
        # Load normal optimized prompts
        normal_cq = PromptLoader.load_optimized_module(
            "prompts/code_quality_normal.json",
            CodeQualityPredictor
        )
        normal_faith = PromptLoader.load_optimized_module(
            "prompts/faithfulness_normal.json",
            FaithfulnessPredictor
        )
        
        if normal_cq:
            prompts['code_quality']['normal'] = normal_cq
        if normal_faith:
            prompts['faithfulness']['normal'] = normal_faith
        
        return prompts


class TestRunner:
    """Runs tests on test dataset."""
    
    def __init__(self, sample_size=None):
        self.prompts = PromptLoader.load_all_prompts()
        self.test_data = self._load_test_data(sample_size)
    
    def _load_test_data(self, sample_size=None):
        """Load the test dataset and optionally sample from it."""
        df = pd.read_excel("data/test_data.xlsx")
        
        if sample_size is not None and sample_size < len(df):
            # Sample random rows from the dataset
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} random examples from {len(pd.read_excel('data/test_data.xlsx'))} total examples")
        
        return df
    
    def evaluate_code_quality(self, module, function_code):
        """Evaluate code quality using given module."""
        with dspy.context(lm=settings.llm_client):
            result = module(function=function_code)
            return float(result.score)
    
    def evaluate_faithfulness(self, module, function_code, requirement):
        """Evaluate faithfulness using given module."""
        with dspy.context(lm=settings.llm_client):
            result = module(function=function_code, requirement=requirement)
            return float(result.score)

    def score_to_category(self, code_quality_score, faithfulness_score, threshold=7):
        """Convert scores to high/low category."""
        if code_quality_score + faithfulness_score >= threshold:
            return "high"
        else:
            return "low"


    def calculate_metrics(self, y_true, y_pred):
        """Calculate precision, recall, specificity, and accuracy manually."""
        # Convert to binary classification metrics for "high" as positive class
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == "high" and pred == "high")
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == "low" and pred == "high")
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == "high" and pred == "low")
        tn = sum(1 for true, pred in zip(y_true, y_pred) if true == "low" and pred == "low")
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def run_test(self):
        """Run test on all prompts and print results."""
        print("="*80)
        print("CODE QUALITY PROMPT TESTING")
        print("="*80)
        print(f"Testing {len(self.test_data)} examples")
        print()
        
        # Get ground truth categories
        y_true = self.test_data['quality'].str.lower().tolist()
        
        # Test each prompt variant
        all_results = {}
        
        for prompt_type in ['original', 'normal', 'bayesian']:
            print(f"Testing {prompt_type} prompt...")
            
            code_quality_module = self.prompts['code_quality'][prompt_type]
            faithfulness_module = self.prompts['faithfulness'][prompt_type]
            predictions = []
            code_quality_scores = []
            faithfulness_scores = []
            
            # Evaluate each code sample
            for _, row in self.test_data.iterrows():
                code = row['code']
                requirement = row['Question']
                code_quality_score = self.evaluate_code_quality(code_quality_module, code)
                faithfulness_score = self.evaluate_faithfulness(faithfulness_module, code, requirement)
                category = self.score_to_category(code_quality_score, faithfulness_score)

                code_quality_scores.append(code_quality_score)
                faithfulness_scores.append(faithfulness_score)
                predictions.append(category)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, predictions)
            
            all_results[prompt_type] = {
                'code_quality_scores': code_quality_scores,
                'faithfulness_scores': faithfulness_scores,
                'predictions': predictions,
                'metrics': metrics
            }
        
        # Print results table
        self._print_results(all_results, y_true)
        
        # Save results
        self._save_results(all_results, y_true)
        
        return all_results
    
    def _print_results(self, results, y_true):
        """Print comparison results."""
        print("\\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        # Print metrics table
        print(f"{'Prompt':<15} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'Accuracy':<12}")
        print("-" * 67)
        
        for prompt_type in ['original', 'normal', 'bayesian']:
            metrics = results[prompt_type]['metrics']
            print(f"{prompt_type.capitalize():<15} "
                  f"{metrics['precision']:<12.3f} "
                  f"{metrics['recall']:<12.3f} "
                  f"{metrics['specificity']:<12.3f} "
                  f"{metrics['accuracy']:<12.3f}")
    
    def _save_results(self, results, y_true):
        """Save test results to artefacts folder."""
        # Prepare data for saving
        detailed_results = []
        
        for i in range(len(y_true)):
            detailed_results.append({
                'index': i,
                'true_category': y_true[i],
                'original_code_quality_score': results['original']['code_quality_scores'][i],
                'normal_code_quality_score': results['normal']['code_quality_scores'][i],
                'bayesian_code_quality_score': results['bayesian']['code_quality_scores'][i],
                'original_faithfulness_score': results['original']['faithfulness_scores'][i],
                'normal_faithfulness_score': results['normal']['faithfulness_scores'][i],
                'bayesian_faithfulness_score': results['bayesian']['faithfulness_scores'][i],
                'original_prediction': results['original']['predictions'][i],
                'normal_prediction': results['normal']['predictions'][i],
                'bayesian_prediction': results['bayesian']['predictions'][i]
            })
        
        # Summary metrics
        summary_metrics = {}
        for prompt_type in ['original', 'normal', 'bayesian']:
            summary_metrics[prompt_type] = results[prompt_type]['metrics']
        
        # Save to JSON
        output_data = {
            'summary_metrics': summary_metrics,
            'detailed_results': detailed_results,
            'metadata': {
                'total_samples': len(y_true),
                'score_threshold': 7,
                'positive_class': 'high'
            }
        }
        
        filename = "artefacts/test_results_1.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\\nResults saved to: {filename}")

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Run code quality and faithfulness tests')
    parser.add_argument('--sample-size', type=int, default=None, 
                        help='Number of random samples to use from test data (default: use all data)')
    
    args = parser.parse_args()
    
    tester = TestRunner(sample_size=args.sample_size)
    tester.run_test()


if __name__ == "__main__":
    main()
