from promptflow import tool
from utils import is_valid_string
from promptflow._utils.logger_utils import logger
from constants import Metric, QAField, MetricGroup


def validate_metric_input(input_data_validation: dict,
                          required_fields: set,
                          metric_name: str
                          ) -> dict:
    for field in required_fields:
        if not input_data_validation[field]:
            logger.error("Input %s is not valid for %s evaluation."
                         % (field, metric_name))
            return False
    return True


@tool
def validate_input(question: str,
                   answer: str,
                   context: str,
                   ground_truth: str,
                   use_qr: str,
                   query: str,
                   response: str,
                   selected_metrics: dict) -> dict:
    input_data_validation = {
        QAField.QUESTION: is_valid_string(question),
        QAField.ANSWER: is_valid_string(answer),
        QAField.CONTEXT: is_valid_string(context),
        QAField.GROUND_TRUTH: is_valid_string(ground_truth),
        QAField.QUERY: is_valid_string(query),
        QAField.RESPONSE: is_valid_string(response)
    }

    if use_qr == "false":
        dict_metric_required_fields = {
            Metric.GPTGroundedness: set(
                [QAField.ANSWER,
                QAField.CONTEXT]),
            Metric.GPTRelevance: set(
                [QAField.QUESTION,
                QAField.ANSWER]),
            Metric.GPTCoherence: set(
                [QAField.QUESTION, QAField.ANSWER]),
            Metric.GPTSimilarity: set(
                [QAField.QUESTION,
                QAField.ANSWER,
                QAField.GROUND_TRUTH]),
            Metric.GPTFluency: set(
                [QAField.ANSWER]),
            Metric.F1Score: set(
                [QAField.ANSWER, QAField.GROUND_TRUTH]),
            Metric.BleuScore: set(
                [QAField.ANSWER, QAField.GROUND_TRUTH]),
            Metric.GleuScore: set(
                [QAField.ANSWER, QAField.GROUND_TRUTH]),
            Metric.MeteorScore: set(
                [QAField.ANSWER, QAField.GROUND_TRUTH]),
            Metric.RougeScore: set(
                [QAField.ANSWER, QAField.GROUND_TRUTH]),
            Metric.HateFairness: set([QAField.QUESTION, QAField.ANSWER]),
            Metric.SelfHarm: set([QAField.QUESTION, QAField.ANSWER]),
            Metric.Sexual: set([QAField.QUESTION, QAField.ANSWER]),
            Metric.Violence: set([QAField.QUESTION, QAField.ANSWER]),
            Metric.ProtectedMaterial: set([QAField.QUESTION, QAField.ANSWER]),
            Metric.IndirectAttack: set([QAField.QUESTION, QAField.ANSWER]),
            Metric.ECI: set([QAField.QUESTION, QAField.ANSWER]),
        }
    else:
        dict_metric_required_fields = {
            Metric.GPTGroundedness: set(
                [QAField.RESPONSE,
                QAField.CONTEXT]),
            Metric.GPTRelevance: set(
                [QAField.QUERY,
                QAField.RESPONSE]),
            Metric.GPTCoherence: set(
                [QAField.QUERY, QAField.RESPONSE]),
            Metric.GPTSimilarity: set(
                [QAField.QUERY,
                QAField.RESPONSE,
                QAField.GROUND_TRUTH]),
            Metric.GPTFluency: set(
                [QAField.RESPONSE]),
            Metric.F1Score: set(
                [QAField.RESPONSE, QAField.GROUND_TRUTH]),
            Metric.BleuScore: set(
                [QAField.RESPONSE, QAField.GROUND_TRUTH]),
            Metric.GleuScore: set(
                [QAField.RESPONSE, QAField.GROUND_TRUTH]),
            Metric.MeteorScore: set(
                [QAField.RESPONSE, QAField.GROUND_TRUTH]),
            Metric.RougeScore: set(
                [QAField.RESPONSE, QAField.GROUND_TRUTH]),
            Metric.HateFairness: set([QAField.QUERY, QAField.RESPONSE]),
            Metric.SelfHarm: set([QAField.QUERY, QAField.RESPONSE]),
            Metric.Sexual: set([QAField.QUERY, QAField.RESPONSE]),
            Metric.Violence: set([QAField.QUERY, QAField.RESPONSE]),
            Metric.ProtectedMaterial: set([QAField.QUERY, QAField.RESPONSE]),
            Metric.IndirectAttack: set([QAField.QUERY, QAField.RESPONSE]),
            Metric.ECI: set([QAField.QUERY, QAField.RESPONSE]),
        }

    data_validation = {}
    for metric_name, required_fields in dict_metric_required_fields.items():
        required_fields = dict_metric_required_fields[metric_name]
        data_validation[metric_name] = False
        if metric_name in selected_metrics[MetricGroup.QUALITY_METRICS]:
            if selected_metrics[MetricGroup.QUALITY_METRICS][metric_name]:
                data_validation[metric_name] = validate_metric_input(
                    input_data_validation,
                    required_fields,
                    metric_name)
            else:
                logger.info("%s is not selected." % metric_name)
        elif metric_name in selected_metrics[MetricGroup.SAFETY_METRICS]:
            if selected_metrics[MetricGroup.SAFETY_METRICS][metric_name]:
                data_validation[metric_name] = validate_metric_input(
                    input_data_validation,
                    required_fields,
                    metric_name
                )
            else:
                logger.info("%s is not selected." % metric_name)
        elif metric_name in selected_metrics[MetricGroup.LEGAL_SECURITY_METRICS]:
            if selected_metrics[MetricGroup.LEGAL_SECURITY_METRICS][metric_name]:
                data_validation[metric_name] = validate_metric_input(
                    input_data_validation,
                    required_fields,
                    metric_name
                )
            else:
                logger.info("%s is not selected." % metric_name)
    return data_validation
