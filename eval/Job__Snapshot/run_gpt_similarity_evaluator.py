from promptflow import tool
from promptflow.evals.evaluators import SimilarityEvaluator
from promptflow.entities import AzureOpenAIConnection
from promptflow.core import AzureOpenAIModelConfiguration


@tool
def run_gpt_similarity(connection: AzureOpenAIConnection, deployment_name, question, answer, ground_truth, query: str, response: str, use_qr: str):
    if use_qr == "true":
        answer = response
        question = query
    inputs = {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
    }

    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=connection.api_base,
        api_key=connection.api_key,
        api_version=connection.api_version,
        azure_deployment=deployment_name
    )

    eval_fn = SimilarityEvaluator(model_config)
    return eval_fn(**inputs)