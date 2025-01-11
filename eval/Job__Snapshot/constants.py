from enum import Enum


class RAIService:
    """Define constants related to RAI service"""
    TIMEOUT = 1800
    SLEEPTIME = 2
    HARM_SEVERITY_THRESHOLD = 4


class Metric:
    """Defines all metrics supported by RAI service"""
    Metrics = "metrics"

    # Content harm
    SelfHarm = "self_harm"
    Violence = "violence"
    Sexual = "sexual"
    HateFairness = "hate_unfairness"
    GPTGroundedness = "gpt_groundedness"
    GPTRelevance = "gpt_relevance"
    GPTSimilarity = "gpt_similarity"
    GPTCoherence = "gpt_coherence"
    GPTFluency = "gpt_fluency"
    F1Score = "f1_score"
    BleuScore = "bleu_score"
    GleuScore = "gleu_score"
    MeteorScore = "meteor_score"
    RougeScore = "rouge_score"
    RougeF1Score = "rouge_f1_score"
    RougeRecall = "rouge_recall"
    RougePrecision = "rouge_precision"
    ProtectedMaterial = "protected_material"
    IndirectAttack = "xpia"
    ECI = "eci"

    LEGAL_SECURITY_METRICS = {
        ProtectedMaterial,
        IndirectAttack,
        ECI,
    }

    QUALITY_METRICS = {
        GPTGroundedness,
        GPTSimilarity,
        GPTFluency,
        GPTCoherence,
        GPTRelevance,
        F1Score,
        BleuScore,
        GleuScore,
        MeteorScore,
        RougeScore,
        RougeF1Score,
        RougeRecall,
        RougePrecision
        }

    # Content harm metric set
    CONTENT_HARM_METRICS = {
        SelfHarm,
        Violence,
        Sexual,
        HateFairness
    }


class MetricGroup:
    """Defines metric groups supported by RAI Service."""
    QUALITY_METRICS = "quality_metrics"
    SAFETY_METRICS = "safety_metrics"
    LEGAL_SECURITY_METRICS = "legal_security_metrics"


class MetricRange:
    """Defines the range of metrics"""
    SAFETY_METRICS_MIN = 0
    SAFETY_METRICS_MAX = 7
    QUALITY_METRICS_MIN = 1
    QUALITY_METRICS_MAX = 5


class HarmSeverityLevel(Enum):
    VeryLow = "Very low"
    Low = "Low"
    Medium = "Medium"
    High = "High"


class Service:
    """Defines supported annotation services."""
    ContentHarm = "content harm"
    Groundedness = "groundedness"


class ServiceVersion:
    """Defines prompt version of supported annotation services."""
    ContentHarm = "0.3"


class QAField:
    QUESTION = "question"
    ANSWER = "answer"
    CONTEXT = "context"
    GROUND_TRUTH = "ground_truth"
    QUERY = "query"
    RESPONSE = "response"
