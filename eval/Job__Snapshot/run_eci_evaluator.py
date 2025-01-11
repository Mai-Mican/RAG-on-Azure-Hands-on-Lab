from promptflow import tool
from azure.identity import DefaultAzureCredential
from promptflow.evals.evaluators._eci._eci import ECIEvaluator as evaluator
from utils import get_current_project_scope


@tool
def run_eci_evaluator(question, answer, query: str, response: str, use_qr: str):
    project_scope = get_current_project_scope()
    eval_fn = evaluator(project_scope, DefaultAzureCredential())
    if use_qr == "true":
        answer = response
        question = query
    return eval_fn(
        question=question,
        answer=answer,
    )
