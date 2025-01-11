from promptflow import tool
from promptflow.entities import AzureOpenAIConnection
from azure.ai.evaluation import RelevanceEvaluator, AzureOpenAIModelConfiguration


@tool
def run_gpt_relevance_evaluator(connection: AzureOpenAIConnection, deployment_name, answer, question, query: str, response: str, use_qr: str):
    if use_qr == "false":
        response = answer
        query = question
    inputs = {
        "response": response,
        "query": query,
    }
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=connection.api_base,
        api_key=connection.api_key,
        api_version=connection.api_version,
        azure_deployment=deployment_name
    )

    eval_fn = RelevanceEvaluator(model_config)
    return eval_fn(**inputs)