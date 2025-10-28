import json
import pandas as pd
import dspy
import matplotlib.pyplot as plt
import numpy as np
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
        
        self._save_results(all_results)
        self._create_validation_plots(all_results)
        return all_results
    
    def _save_results(self, results):
        """Save validation results to artefacts folder."""
        filename = "artefacts/validation_results.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
    def _create_validation_plots(self, results):
        """Create line plots comparing prompt performance with human annotations."""
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = {
            'original': '#1f77b4',    
            'normal': '#ff7f0e',      
            'bayesian': '#2ca02c'     
        }
        
        metrics = ['code_quality', 'faithfulness']
        axes = [ax1, ax2]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get question IDs and sort them
            question_ids = sorted(results[metric].keys())
            x_positions = range(len(question_ids))
            
            # Plot lines for each prompt type
            for prompt_type in ['original', 'normal', 'bayesian']:
                y_values = [results[metric][qid][prompt_type] for qid in question_ids]
                ax.plot(x_positions, y_values, 
                       color=colors[prompt_type], 
                       marker='o', 
                       linewidth=2, 
                       markersize=6,
                       label=prompt_type.capitalize())
            
            # Add human annotations as black dots
            self._add_human_annotations(ax, metric, question_ids, x_positions)
            
            # Customize the plot
            ax.set_xlabel('Question ID')
            ax.set_ylabel('Score')
            ax.set_title(f'{metric.replace("_", " ").title()} Validation Results')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(question_ids)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 6)  
        
        # Adjust layout and save
        plt.tight_layout()
        plot_filename = "artefacts/validation_plots.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            
    def _add_human_annotations(self, ax, metric, question_ids, x_positions):
        """Add human annotation points to the plot."""
        metric_key = 'code_quality' if metric == 'code_quality' else 'faithfulness'
        
        # Collect all human scores for each question
        for i, question_id in enumerate(question_ids):
            if question_id in self.human_annotations and metric_key in self.human_annotations[question_id]:
                human_scores = []
                for annotator, score in self.human_annotations[question_id][metric_key].items():
                    human_scores.append(score)
                
                # Plot individual human scores as black dots with slight horizontal offset
                if human_scores:
                    # Add small random offsets to avoid overlapping points
                    x_offsets = np.random.normal(0, 0.02, len(human_scores))
                    x_coords = [i + offset for offset in x_offsets]
                    
                    ax.scatter(x_coords, human_scores, 
                             color='black', 
                             alpha=0.7, 
                             s=30, 
                             marker='o',
                             label='Human Annotations' if i == 0 else "")  

def main():
    """Main validation function."""
    validator = ValidationRunner()
    validator.run_validation()


if __name__ == "__main__":
    main()
