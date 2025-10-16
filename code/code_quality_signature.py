import dspy


class EvaluateCodeQuality(dspy.Signature):
    """
    Evaluate the quality of the provided function. Evaluate the code quality based on 
    metrics like readability, maintainability, efficiency, and adherence to best practices.

    Score based on these specifications:
    - 1 = Poor quality: The code is difficult to read, understand, and maintain. It may contain
      inefficient algorithms, lack of comments, and poor structure.
    - 2 = Below average quality: The code has some readability and maintainability issues. 
      It may be inefficient in parts and lacks proper documentation.
    - 3 = Fair quality: The code is somewhat readable and maintainable but has several
      areas for improvement. It may have some inefficient parts and lacks consistency in style.
    - 4 = Good quality: The code is generally readable and maintainable. It follows
      best practices and has a good structure, but there may be minor inefficiencies or
      areas for improvement.
    - 5 = Excellent quality: The code is highly readable, maintainable, and efficient.
      It adheres to best practices, has a clear structure, and is well-documented.
    
    The return value of this function should always be an continuous number between 1 and 5.
    """

    function: str = dspy.InputField(description="Function to be evaluated.")
    score: float = dspy.OutputField(
        description="A float representing the code quality score of the function."
    )
