from promptflow import tool
from promptflow.evals.evaluators import MeteorScoreEvaluator


@tool
def compute_meteor_score(meteor_params: dict, ground_truth: str, answer: str, response: str, use_qr: str) -> str:
    import nltk
    try:
        nltk.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')

    eval_fun = MeteorScoreEvaluator(
        alpha=meteor_params.get("alpha", 0.9),
        beta=meteor_params.get("beta", 3.0),
        gamma=meteor_params.get("gamma", 0.5))

    if use_qr == "true":
        answer = response
    result = eval_fun(answer=answer, ground_truth=ground_truth)
    return result
