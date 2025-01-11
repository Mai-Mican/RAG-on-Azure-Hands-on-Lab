from promptflow import tool
from promptflow.entities import AzureOpenAIConnection
from azure.ai.evaluation import FluencyEvaluator, AzureOpenAIModelConfiguration


@tool
def run_gpt_fluency_evaluator(connection: AzureOpenAIConnection, deployment_name, answer, response: str, use_qr: str):
    if use_qr == "false":
        response = answer
    inputs = {
        "response": response,
    }
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=connection.api_base,
        api_key=connection.api_key,
        api_version=connection.api_version,
        azure_deployment=deployment_name
    )

    eval_fn = FluencyEvaluator(model_config)
    return eval_fn(**inputs)
