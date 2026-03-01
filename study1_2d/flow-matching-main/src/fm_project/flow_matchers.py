from typing import Dict


def _import_torchcfm():
    try:
        from torchcfm.conditional_flow_matching import (
            ExactOptimalTransportConditionalFlowMatcher,
            SchrodingerBridgeConditionalFlowMatcher,
            TargetConditionalFlowMatcher,
            VariancePreservingConditionalFlowMatcher,
        )
    except ImportError as e:
        raise ImportError(
            "torchcfm is required. Install with: pip install torchcfm"
        ) from e
    return {
        "ExactOptimalTransportConditionalFlowMatcher": ExactOptimalTransportConditionalFlowMatcher,
        "VariancePreservingConditionalFlowMatcher": VariancePreservingConditionalFlowMatcher,
        "TargetConditionalFlowMatcher": TargetConditionalFlowMatcher,
        "SchrodingerBridgeConditionalFlowMatcher": SchrodingerBridgeConditionalFlowMatcher,
    }


def build_matchers(variant_to_class: Dict[str, str], sigma: float):
    classes = _import_torchcfm()
    out = {}
    for variant, class_name in variant_to_class.items():
        if class_name not in classes:
            raise ValueError(f"Unknown flow matcher class '{class_name}'")
        out[variant] = classes[class_name](sigma=sigma)
    return out
