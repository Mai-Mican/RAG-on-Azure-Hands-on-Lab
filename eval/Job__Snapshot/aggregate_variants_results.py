from typing import Dict, List
#from promptflow import tool, log_metric
from promptflow.core import tool, log_metric
import numpy as np
from utils import filter_metrics
from constants import RAIService, MetricGroup, Metric


@tool
def aggregate_results(results: List[Dict[str, str]],
                      selected_metrics: List[Dict[str, bool]],
                      thresholds: List[int]) -> Dict[str, float]:
    if selected_metrics:
        selected_safety_metrics = filter_metrics(
            selected_metrics[0][MetricGroup.SAFETY_METRICS])
        selected_quality_metrics = filter_metrics(
            selected_metrics[0][MetricGroup.QUALITY_METRICS])
        selected_legacy_security_metrics = filter_metrics(
            selected_metrics[0][MetricGroup.LEGAL_SECURITY_METRICS])
        selected_legacy_security_metrics_with_label = [item + "_label" for item in selected_legacy_security_metrics]
    else:
        selected_safety_metrics = []
        selected_quality_metrics = []
        selected_legacy_security_metrics = []

    if Metric.RougeScore in selected_quality_metrics:
        selected_quality_metrics.remove(Metric.RougeScore)
        selected_quality_metrics.extend([
            Metric.RougeF1Score,
            Metric.RougeRecall,
            Metric.RougePrecision
        ])

    if thresholds != [] and thresholds is not None:
        threshold = np.float16(thresholds[0])
    else:
        threshold = np.float16(
            RAIService.HARM_SEVERITY_THRESHOLD)

    aggregate_results = {}
    for result in results:
        if not result:
            continue
        for name in result.keys():
            if name in selected_quality_metrics or \
                name in selected_safety_metrics or \
                name in selected_legacy_security_metrics_with_label:
                if name not in aggregate_results.keys():
                    aggregate_results[name] = []

                metric_value = result[name]
                if name in selected_safety_metrics:
                    metric_value = result[name + "_score"]

                try:
                    float_val = float(metric_value)
                except Exception:
                    float_val = np.nan
                if float_val >= 0:
                    aggregate_results[name].append(float_val)

    aggregate_output = {}
    for name, values in aggregate_results.items():
        metric_name = name
        if name in selected_safety_metrics:
            metric_name = name + '_defect_rate'

        if name in selected_legacy_security_metrics_with_label:
            metric_name = name.replace("_label", "_defect_rate")

        # Filter out NaN values
        values = np.array(values)
        values = values[~np.isnan(values)]

        if len(values) == 0:
            aggregate_output[metric_name] = np.nan
        else:
            if name in selected_quality_metrics:
                aggregate_output[metric_name] = round(np.nanmean(values), 2)
            elif name in selected_safety_metrics:
                aggregate_output[metric_name] = round(
                    np.sum(values >= threshold) / len(values), 2)
            elif name in selected_legacy_security_metrics_with_label:
                aggregate_output[metric_name] = round(
                    np.sum(values) / len(values), 2)
            else:
                aggregate_output[metric_name] = np.nan
        log_metric(metric_name, aggregate_output[metric_name])
    return aggregate_output

# Print the final output
print("Aggregate Output:")
for key, value in aggregate_output.items():
    print(f"{key}: {value}")
