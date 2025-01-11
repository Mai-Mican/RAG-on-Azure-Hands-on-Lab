from typing import Dict

from promptflow import tool
from constants import Metric
import numpy as np


def default_safety_result(metric_name):
    return {
        metric_name: np.nan,
        metric_name + "_score": np.nan,
        metric_name + "_reason": np.nan,
    }


def default_quality_result(metric_name: str):
    if metric_name == Metric.RougeScore:
        return {
            "rouge_f1_score": np.nan,
            "rouge_recall": np.nan,
            "rouge_precision": np.nan,
        }
    metric_reason = f"{metric_name}_reason"
    if metric_reason.startswith("gpt_"):
        metric_reason = metric_reason[4:]
    return {
        metric_name: np.nan,
        metric_reason: np.nan,
    }

def default_legacy_security_result(metric_name):
    result = {
        metric_name + "_label": np.nan,
        metric_name + "_reason": np.nan,
    }

    if metric_name == Metric.IndirectAttack:
        result.update({
            metric_name + "_intrusion": np.nan,
            metric_name + "_information_gathering": np.nan,
            metric_name + "_manipulated_content": np.nan,
        })

    return result


@tool
def concat_results(
    f1_score_results: Dict[str, str] = None,
    bleu_score_results: Dict[str, str] = None,
    gleu_score_results: Dict[str, str] = None,
    meteor_score_results: Dict[str, str] = None,
    rouge_score_results: Dict[str, str] = None,
    gpt_groundedness_results: Dict[str, str] = None,
    gpt_coherence_results: Dict[str, str] = None,
    gpt_fluency_results: Dict[str, str] = None,
    gpt_relevance_results: Dict[str, str] = None,
    gpt_similarity_results: Dict[str, str] = None,
    hate_unfairness_results: Dict[str, str] = None,
    self_harm_results: Dict[str, str] = None,
    sexual_results: Dict[str, str] = None,
    violence_results: Dict[str, str] = None,
    protected_material_results: Dict[str, str] = None,
    xpia_results: Dict[str, str] = None,
    eci_results: Dict[str, str] = None,
) -> Dict[str, str]:
    concated_results = {}

    concated_results.update(f1_score_results or default_quality_result(Metric.F1Score))
    concated_results.update(bleu_score_results or default_quality_result(Metric.BleuScore))
    concated_results.update(gleu_score_results or default_quality_result(Metric.GleuScore))
    concated_results.update(meteor_score_results or default_quality_result(Metric.MeteorScore))
    concated_results.update(rouge_score_results or default_quality_result(Metric.RougeScore))
    concated_results.update(gpt_groundedness_results or default_quality_result(Metric.GPTGroundedness))
    concated_results.update(gpt_coherence_results or default_quality_result(Metric.GPTCoherence))
    concated_results.update(gpt_fluency_results or default_quality_result(Metric.GPTFluency))
    concated_results.update(gpt_relevance_results or default_quality_result(Metric.GPTRelevance))
    concated_results.update(gpt_similarity_results or default_quality_result(Metric.GPTSimilarity))
    concated_results.update(hate_unfairness_results or default_safety_result(Metric.HateFairness))
    concated_results.update(self_harm_results or default_safety_result(Metric.SelfHarm))
    concated_results.update(sexual_results or default_safety_result(Metric.Sexual))
    concated_results.update(violence_results or default_safety_result(Metric.Violence))
    concated_results.update(protected_material_results or default_legacy_security_result(Metric.ProtectedMaterial))
    concated_results.update(xpia_results or default_legacy_security_result(Metric.IndirectAttack))
    concated_results.update(eci_results or default_legacy_security_result(Metric.ECI))
    
    return concated_results
