from promptflow import tool
from promptflow.evals.evaluators import GleuScoreEvaluator


@tool
def compute_gleu_score(ground_truth: str, answer: str, response: str, use_qr: str) -> str:
    eval_fun = GleuScoreEvaluator()
    if use_qr == "true":
        answer = response
    return eval_fun(answer=answer, ground_truth=ground_truth)
