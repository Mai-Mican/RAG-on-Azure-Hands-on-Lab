from promptflow import tool
from promptflow.evals.evaluators import BleuScoreEvaluator


@tool
def compute_bleu_score(ground_truth: str, answer: str, response: str, use_qr: str) -> str:
    eval_fun = BleuScoreEvaluator()
    if use_qr == "true":
        answer = response
    return eval_fun(answer=answer, ground_truth=ground_truth)
