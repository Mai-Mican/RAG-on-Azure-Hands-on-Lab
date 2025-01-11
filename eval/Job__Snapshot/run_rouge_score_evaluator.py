from promptflow import tool
from promptflow.evals.evaluators import RougeScoreEvaluator


@tool
def compute_rouge_score(rouge_type: str, ground_truth: str, answer: str, response: str, use_qr: str) -> str:
    eval_fun = RougeScoreEvaluator(rouge_type=rouge_type)
    if use_qr == "true":
        answer = response
    result = eval_fun(answer=answer, ground_truth=ground_truth)
    return result
