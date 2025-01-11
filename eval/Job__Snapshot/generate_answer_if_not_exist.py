from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from promptflow import tool
from promptflow.connections import AzureOpenAIConnection, CustomConnection
from openai import AzureOpenAI
import re
from typing import Union


def is_valid_string(input_string: str) -> bool:
    # if input_string contains any letter or number,
    # it is a valid string
    if not input_string:
        return False
    return bool(re.search(r"\d|\w", input_string))


@tool
def generate_answer_if_not_exist(
    use_qr: str,
    query: str,
    response: str,
    question: str,
    context: str,
    answer: str,
    connection: Union[AzureOpenAIConnection, CustomConnection] = {},
    deployment_name: str = "",
    model_prompt: object = {},
    endpoint_url: str = "",
) -> bool:
    if use_qr == "true":
        answer = response
        question = query

    if not connection or not deployment_name:
        return answer

    if is_valid_string(answer):
        return answer
    return get_response(
        question,
        context,
        connection,
        deployment_name,
        model_prompt,
        endpoint_url,
    )


def get_response(
    question: str,
    context: str,
    connection: Union[AzureOpenAIConnection, CustomConnection],
    deployment_name: str,
    model_prompt: object,
    endpoint_url: str = "",
) -> str:
    system_message = model_prompt["system_message"]
    few_shot_examples = model_prompt["few_shot_examples"]
    model_type = model_prompt["model_type"]
    temperature = model_prompt["model_params"]["temperature"]
    max_tokens = model_prompt["model_params"]["max_tokens"]
    top_p = model_prompt["model_params"]["top_p"]

    formatted_system_message = [{
        "role": "system",
        "content": system_message
    }]

    formatted_few_shot_examples = []
    for item in few_shot_examples:
        for example in few_shot_examples[item]:
            formatted_few_shot_examples.append(
                {"role": "user", "content": example["user"]}
            )
            formatted_few_shot_examples.append(
                {"role": "assistant", "content": example["assistant"]}
            )

    if context:
        formatted_user_message = [{
            "role": "user",
            "content": f"{context} {question}"
        }]
    else:
        formatted_user_message = [{"role": "user", "content": question}]

    messages = (
        formatted_system_message
        + formatted_few_shot_examples
        + formatted_user_message
    )

    if isinstance(connection, AzureOpenAIConnection):
        if connection.api_key:
            config = {
                # disable OpenAI's built-in retry mechanism by using
                # our own retry for better debuggability and real-time
                # status updates.
                "max_retries": 0,
                "api_key": connection.api_key,
                "api_version": connection.api_version,
                "azure_endpoint": connection.api_base,
            }
        else:
            config = {
                "max_retries": 0,
                "api_version": connection.api_version,
                "azure_endpoint": connection.api_base,
                "azure_ad_token_provider": connection.get_token,
            }

        client = AzureOpenAI(**config)
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        return response.model_dump()["choices"][0]["message"]["content"]

    elif isinstance(connection, CustomConnection):
        if not endpoint_url:
            raise ValueError("endpoint_url is required for custom connection")
        
        api_key = connection[deployment_name]
        client = ChatCompletionsClient(
            endpoint=endpoint_url,
            credential=AzureKeyCredential(api_key),
        )
        
        response = client.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return response.choices[0].message.content

    else:
        raise ValueError("Model type %s is not supported" % (model_type))
