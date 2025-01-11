from promptflow import tool
from promptflow.entities import AzureOpenAIConnection
from azure.ai.evaluation import GroundednessEvaluator, AzureOpenAIModelConfiguration


@tool
def run_gpt_groundedness_evaluator(connection: AzureOpenAIConnection, deployment_name, query, question, answer, context, response: str, use_qr: str):
    if use_qr == "false":
        query = question
        response = answer
    inputs = {
        "response": response,
        "query": query,
        "context": context,
    }
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=connection.api_base,
        api_key=connection.api_key,
        api_version=connection.api_version,
        azure_deployment=deployment_name
    )

    eval_fn = GroundednessEvaluator(model_config)
    return eval_fn(**inputs)