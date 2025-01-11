from promptflow import tool
from promptflow.entities import AzureOpenAIConnection
from azure.ai.evaluation import CoherenceEvaluator, AzureOpenAIModelConfiguration

@tool
def run_gpt_coherence_evaluator(connection: AzureOpenAIConnection, deployment_name, question, answer, query: str, response: str, use_qr: str):
    if use_qr == "false":
        response = answer
        query = question
    inputs = {
        "query": question,
        "response": answer,
    }

    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=connection.api_base,
        api_key=connection.api_key,
        api_version=connection.api_version,
        azure_deployment=deployment_name
    )

    eval_fn = CoherenceEvaluator(model_config)
    return eval_fn(**inputs)