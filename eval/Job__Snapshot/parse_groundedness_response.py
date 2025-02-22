from promptflow import tool
from typing import List
import numpy as np
import re
from constants import Metric


def parse_single_sample(response: dict) -> list:
    parsed_response = []
    if not response:
        return parsed_response
    for key in response:
        harm_type = key.replace("generic", "gpt")
        parsed_harm_response = {}
        try:
            harm_response = eval(response[key])
        except Exception:
            harm_response = response[key]
        if harm_response != "" and isinstance(harm_response, dict):
            # check if "output" is one key in harm_response
            if "output" in harm_response:
                harm_response = harm_response["output"]

            # get content harm metric_value
            if 'label' in harm_response:
                metric_value = harm_response['label']
            else:
                metric_value = np.nan

            # get reasoning
            if "reasoning" in harm_response:
                reasoning = harm_response['reasoning']
            elif "reason" in harm_response:
                reasoning = harm_response['reason']
            else:
                reasoning = ""
        elif harm_response != "" and isinstance(harm_response, str):
            metric_value_match = re.findall(r"(\b[0-7])\b", harm_response)
            if metric_value_match:
                metric_value = int(metric_value_match[0])
            else:
                metric_value = np.nan
            reasoning = harm_response
        elif harm_response != "" and (isinstance(harm_response, int)
                                      or isinstance(harm_response, float)):
            if harm_response >= 0 and harm_response <= 7:
                metric_value = harm_response
            else:
                metric_value = np.nan
            reasoning = ""
        else:
            metric_value = np.nan
            reasoning = ""
        parsed_harm_response[harm_type] = metric_value
        parsed_harm_response[harm_type + "_reason"] = reasoning
        parsed_response.append(parsed_harm_response)
    return parsed_response

@tool
def parse_response(is_service_available: dict,
                   batch_response: List[dict] = None):
    parsed_single_sample_response = None
    if is_service_available["groundedness_service"]:
        if batch_response:
            single_sample_response = batch_response[0]
            parsed_single_sample_response = parse_single_sample(
                single_sample_response)[0]

    return parsed_single_sample_response
