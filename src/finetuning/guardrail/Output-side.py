# ---- Output Guardrail: Non-factual / unsupported answer filter ---------------
import os, re
from typing import List, Dict, Any, Tuple
from transformers import pipeline

NLI_MODEL = "typeform/distilroberta-base-mnli"  # small, fast, good enough
ENTAILMENT_MIN = 0.60   # require at least this entailment prob
CONTRADICTION_MAX = 0.25  # and keep contradiction below this
SNIPPET_WINDOW = 320    # chars around answer when available

_numeral_re = re.compile(r"\d+(?:\.\d+)?%?")

def _build_nli():
    use_gpu = os.environ.get("USE_GPU", "").strip() == "1"
    device = 0 if use_gpu else -1
    return pipeline("text-classification", model=NLI_MODEL, device=device)

def _extract_numbers(s: str) -> List[str]:
    return _numeral_re.findall(s or "")

def _snippet_around(text: str, answer: str, w: int = SNIPPET_WINDOW) -> str:
    i = text.lower().find((answer or "").lower())
    if i < 0:
        # fall back to a dense middle slice to keep premise short
        mid = max(0, len(text) // 2 - w // 2)
        return text[mid:mid + w]
    start = max(0, i - w // 2)
    end = min(len(text), i + len(answer) + w // 2)
    return text[start:end]

def _nli_scores(nli, premise: str, hypothesis: str) -> Dict[str, float]:
    out = nli({"text": premise, "text_pair": hypothesis}, return_all_scores=True)[0]
    # output labels typically: CONTRADICTION / NEUTRAL / ENTAILMENT
    m = {d["label"].lower(): float(d["score"]) for d in out}
    return {
        "entailment": m.get("entailment", 0.0),
        "neutral": m.get("neutral", 0.0),
        "contradiction": m.get("contradiction", 0.0),
    }

def verify_answer_support(
    question: str,
    answer: str,
    contexts: List[str],
    nli=None,
) -> Dict[str, Any]:
    """
    Returns a verdict dict:
      {
        "action": "pass" | "flag",
        "message": str,
        "support_snippet": str,
        "scores": {"entailment": float, "contradiction": float},
        "reasons": [codes...],
      }
    """
    reasons = []
    if not (answer and answer.strip()):
        return {"action": "flag", "message": "Empty answer.", "support_snippet": "", "scores": {}, "reasons": ["empty"]}

    # Quick lexical checks
    answer_in_context = any(answer.lower() in (c or "").lower() for c in contexts)
    nums = _extract_numbers(answer)
    nums_supported = True
    if nums:
        joined = " ".join(contexts).lower()
        nums_supported = all(n.lower() in joined for n in nums)

    if not answer_in_context:
        reasons.append("answer_not_in_context")
    if not nums_supported:
        reasons.append("numeric_unattributed")

    # NLI support: choose best supporting snippet across contexts
    nli = nli or _build_nli()
    hypothesis = f"The answer to the question '{question}' is: {answer}"

    best_ent, best_contra, best_snip = 0.0, 1.0, ""
    for ctx in contexts:
        if not ctx:
            continue
        prem = _snippet_around(ctx, answer, SNIPPET_WINDOW)
        scores = _nli_scores(nli, prem, hypothesis)
        if scores["entailment"] > best_ent:
            best_ent, best_contra, best_snip = scores["entailment"], scores["contradiction"], prem

    if best_ent < ENTAILMENT_MIN:
        reasons.append("low_entailment")
    if best_contra > CONTRADICTION_MAX:
        reasons.append("high_contradiction")

    if reasons:
        return {
            "action": "flag",
            "message": "Output appears insufficiently supported by retrieved context.",
            "support_snippet": best_snip,
            "scores": {"entailment": best_ent, "contradiction": best_contra},
            "reasons": reasons,
        }

    return {
        "action": "pass",
        "message": "Output attribution check passed.",
        "support_snippet": best_snip,
        "scores": {"entailment": best_ent, "contradiction": best_contra},
        "reasons": [],
    }
