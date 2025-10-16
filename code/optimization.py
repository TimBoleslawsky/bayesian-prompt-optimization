import pandas as pd
import json
import dspy
from dspy.teleprompt import MIPROv2

from bayesian_prompt_optimization.settings import settings
from bayesian_prompt_optimization.code.code_quality_signature import EvaluateCodeQuality
from bayesian_prompt_optimization.code.faithfulness_signature import EvaluateFaithfulness


class CodeQualityPredictor(dspy.Module):
    """Module that uses the EvaluateCodeQuality signature."""
    
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict(EvaluateCodeQuality)
    
    def forward(self, function):
        return self.evaluate(function=function)

class FaithfulnessPredictor(dspy.Module):
    """Module that uses the EvaluateFaithfulness signature."""
    
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict(EvaluateFaithfulness)
    
    def forward(self, function, requirement):
        return self.evaluate(function=function, requirement=requirement)

class SignatureOptimizer:
    def optimize_signature(
        self, signature_name, program_class, metric, trainset, minibatch_size
    ):
        with dspy.context(lm=settings.llm_client):
            # Create an instance of the program
            program = program_class()
            teleprompter = MIPROv2(metric=metric)
            optimized_program = teleprompter.compile(
                program,
                trainset=trainset,
                minibatch_size=minibatch_size,
            )
            # Save the optimized program
            optimized_program.save(
                "bayesian_prompt_optimization/prompts/" + signature_name + ".json"
            )
        return optimized_program

def metric(example, pred):
    """
    Metric for continuous score evaluation between 1-5.
    Uses inverted MSE so higher values indicate better performance.
    """
    pred_score = float(pred.score)
    ref_score = float(example.score)
    mse = (pred_score - ref_score) ** 2
    return 1.0 / (1.0 + mse)

def create_dspy_examples_code_quality(df):
    """Convert DataFrame to DSPy examples for code quality evaluation."""
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            function=row['response'],
            score=row['score']
        ).with_inputs('function')
        examples.append(example)
    return examples

def create_dspy_examples_faithfulness(df):
    """Convert DataFrame to DSPy examples for faithfulness evaluation."""
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            function=row['response'],
            requirement=row['requirement'],
            score=row['score']
        ).with_inputs('function', 'requirement')
        examples.append(example)
    return examples
    
if __name__ == "__main__":
    with open("bayesian_prompt_optimization/data/benchmark.json", "r") as f:
        benchmark_data = json.load(f)
    raw = pd.DataFrame(benchmark_data["benchmark"])

    # Load means (strip leading zeros from json ids to match csv ids)
    means = pd.read_csv("bayesian_prompt_optimization/artefacts/means.csv")

    # Normalize IDs for safe merge
    raw["question_id_norm"] = raw["id"].astype(str).str.lstrip("0")
    means["question_id_norm"] = means["question_id"].astype(str)

    # Merge
    merged = raw.merge(means, on="question_id_norm", how="inner", suffixes=("", "_mean_src"))

    # Columns containing mean values
    mean_cols = [
        "posterior_expected_code_quality",
        "posterior_expected_faithfulness",
        "normal_code_quality_means",
        "normal_faithfulness_means",
    ]

    # Base columns to keep (question/answer + original id)
    base_cols = [c for c in raw.columns if c != "question_id_norm"]

    # Build four datasets: each keeps base cols + one mean column (renamed to 'score')
    posterior_code_quality_dataset = merged[base_cols + ["posterior_expected_code_quality"]].rename(
        columns={"posterior_expected_code_quality": "score"}
    )
    posterior_faithfulness_dataset = merged[base_cols + ["posterior_expected_faithfulness"]].rename(
        columns={"posterior_expected_faithfulness": "score"}
    )
    normal_code_quality_dataset = merged[base_cols + ["normal_code_quality_means"]].rename(
        columns={"normal_code_quality_means": "score"}
    )
    normal_faithfulness_dataset = merged[base_cols + ["normal_faithfulness_means"]].rename(
        columns={"normal_faithfulness_means": "score"}
    )

    # optimize signatures
    optimizer = SignatureOptimizer()
    
    # Convert DataFrames to DSPy examples
    posterior_faithfulness_examples = create_dspy_examples_faithfulness(posterior_faithfulness_dataset)
    posterior_code_quality_examples = create_dspy_examples_code_quality(posterior_code_quality_dataset)
    normal_faithfulness_examples = create_dspy_examples_faithfulness(normal_faithfulness_dataset)
    normal_code_quality_examples = create_dspy_examples_code_quality(normal_code_quality_dataset)
    
    optimizer.optimize_signature(
        signature_name="faithfulness_bayesian",
        program_class=FaithfulnessPredictor,
        metric=metric,
        trainset=posterior_faithfulness_examples,
        minibatch_size=5,
    )

    optimizer.optimize_signature(
        signature_name="code_quality_bayesian",
        program_class=CodeQualityPredictor,
        metric=metric,
        trainset=posterior_code_quality_examples,       
        minibatch_size=5,
    )  

    optimizer.optimize_signature(
        signature_name="faithfulness_normal",
        program_class=FaithfulnessPredictor,
        metric=metric,
        trainset=normal_faithfulness_examples,
        minibatch_size=5,
    )

    optimizer.optimize_signature(
        signature_name="code_quality_normal",
        program_class=CodeQualityPredictor,
        metric=metric,
        trainset=normal_code_quality_examples,
        minibatch_size=5,
    )    
