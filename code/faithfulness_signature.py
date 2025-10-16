import dspy


class EvaluateFaithfulness(dspy.Signature):
    """
    Performs static code analysis on the function. Only evaluate function based
    on faithfulness given the requirement as specification. Evaluate
    how well the function represents the requirement.

    Score based on these specifications:
    - 1 = Function does not represent the logic of the requirement at all.
    - 2 = Function barely represents the logic of the requirement.
    - 3 = Function partially represents the logic of the requirement.
    - 4 = Function correctly represents the logic of the requirement, but has some
    semantic differences or interprets the logic slightly differently than the requirement.
    - 5 = Function perfectly represents the logic and semantic of the requirement.

    The return value of this function should always be an continuous number between 1 and 5.
    """

    function: str = dspy.InputField(description="Function to be evaluated.")
    requirement: str = dspy.InputField(
        description="Requirement that serves as specification for the function"
        + "and is basis for the analysis for faithfulness."
    )
    score: float = dspy.OutputField(
        description="A float representing the faithfulness score of the function."
    )
