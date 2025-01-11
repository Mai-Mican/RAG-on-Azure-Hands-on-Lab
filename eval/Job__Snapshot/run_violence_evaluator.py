from promptflow import tool
from azure.identity import DefaultAzureCredential
from promptflow.evals.evaluators import ViolenceEvaluator as evaluator
from utils import get_current_project_scope


@tool
def run_violence_evaluator(question, answer, query: str, response: str, use_qr: str):
    project_scope = get_current_project_scope()
    eval_fn = evaluator(project_scope, DefaultAzureCredential())

    if use_qr == "true":
        answer = response
        question = query

    result = eval_fn(
        question=question,
        answer=answer,
    )
    return result
