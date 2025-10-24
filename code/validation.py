import json
import pandas as pd
import dspy
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


class ValidationRunner:
    """Runs validation tests on holdout dataset."""
    
    def __init__(self):
        self.prompts = PromptLoader.load_all_prompts()
        self.holdout_data = self._load_holdout_data()
        self.human_annotations = self._load_human_annotations()
    
    def _load_holdout_data(self):
        """Load the holdout dataset."""
        with open("data/benchmark_holdout.json", "r") as f:
            data = json.load(f)
        return pd.DataFrame(data["benchmark"])
    
    def _load_human_annotations(self):
        """Load and process human annotations."""
        with open("data/scores_holdout.json", "r") as f:
            annotations = json.load(f)
        
        # Organize scores by question ID and annotator
        human_scores = {}
        for annotator, scores in annotations.items():
            for score_entry in scores:
                question_id = score_entry["id"]
                if question_id not in human_scores:
                    human_scores[question_id] = {
                        'code_quality': {},
                        'faithfulness': {}
                    }
                
                human_scores[question_id]['code_quality'][annotator] = score_entry["score_codequality"]
                human_scores[question_id]['faithfulness'][annotator] = score_entry["score_faithfulness"]
        
        return human_scores
    
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
    
    def run_validation(self):
        """Run validation on all prompts and print results in comparison format."""
        print("="*120)
        print("VALIDATION RESULTS ON HOLDOUT DATASET")
        print("="*120)
        
        # Get annotator names for header
        annotators = list(next(iter(self.human_annotations.values()))['code_quality'].keys())
        
        # Evaluate all prompts for all questions
        all_results = {}
        
        for metric in ['code_quality', 'faithfulness']:
            all_results[metric] = {}
            
            for prompt_type in ['original', 'normal', 'bayesian']:
                module = self.prompts[metric][prompt_type]
                
                for _, row in self.holdout_data.iterrows():
                    question_id = row['id']
                    
                    if question_id not in all_results[metric]:
                        all_results[metric][question_id] = {}
                    
                    if metric == 'code_quality':
                        score = self.evaluate_code_quality(module, row['response'])
                    else:
                        score = self.evaluate_faithfulness(module, row['response'], row['requirement'])
                    
                    all_results[metric][question_id][prompt_type] = score
        
        # Print comparison tables
        for metric in ['code_quality', 'faithfulness']:
            print(f"\n{metric.upper().replace('_', ' ')} EVALUATION")
            print("="*120)
            
            # Create header with all annotators
            header = f"{'ID':<5} {'Original':<10} {'Normal':<10} {'Bayesian':<10}"
            for annotator in annotators:
                header += f" {annotator.strip():<10}"
            print(header)
            print("-" * 120)
            
            # Data rows
            for _, row in self.holdout_data.iterrows():
                question_id = row['id']
                
                # Get LLM scores
                original_score = all_results[metric][question_id]['original']
                normal_score = all_results[metric][question_id]['normal']
                bayesian_score = all_results[metric][question_id]['bayesian']
                
                # Get human scores
                human_scores = self.human_annotations[question_id][metric]
                
                # Format row
                row_str = f"{question_id:<5} {original_score:<10.2f} {normal_score:<10.2f} {bayesian_score:<10.2f}"
                for annotator in annotators:
                    score = human_scores[annotator]
                    row_str += f" {score:<10.0f}"
                
                print(row_str)
        
        self._save_results(all_results)
        return all_results
    
    def _save_results(self, results):
        """Save validation results to artefacts folder."""
        filename = "artefacts/validation_results.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")

def main():
    """Main validation function."""
    validator = ValidationRunner()
    validator.run_validation()


if __name__ == "__main__":
    main()
