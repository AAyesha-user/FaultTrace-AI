"""
FaultTrace AI - AI explanation helper (OpenRouter) - version 2
--------------------------------------------------------------
Save this file as:  backend/ai/explainer.py   (REPLACES the old one)
Also keep an EMPTY file:  backend/ai/__init__.py
 
What it does:
  1. Tries your models one by one (model fallback).
  2. If every model fails, returns an offline summary, so the demo never breaks.
  3. Always tells you WHY it failed (result["error"]), so debugging is easy.
 
The AI only EXPLAINS the diagnosis. Your own code still calculates it.
"""
 
import os
 
import pandas as pd
import requests
 
# OpenRouter speaks the same format as OpenAI. To use another provider
# (Groq, Gemini's OpenAI-compatible endpoint, OpenAI itself...), change this
# URL + the key + the model names. Nothing else needs to change.
API_URL = os.environ.get(
    "AI_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
 
# Tried in this order. Check openrouter.ai/models that these still exist,
# and replace them with any model ids you prefer.
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "openrouter/auto",
]
 
SCORE_COLUMNS = ["consensus_score", "suspiciousness", "diagnostic_score"]
 
 
def get_api_key(explicit_key=None):
    return explicit_key or os.environ.get("OPENROUTER_API_KEY")
 
 
def get_models(explicit=None):
    """Accepts a list, or a comma separated string, or reads OPENROUTER_MODELS."""
    raw = explicit or os.environ.get("OPENROUTER_MODELS")
    if not raw:
        return list(DEFAULT_MODELS)
    if isinstance(raw, str):
        raw = raw.split(",")
    models = [m.strip() for m in raw if m and m.strip()]
    return models or list(DEFAULT_MODELS)
 
 
def _score_column(diagnosis: pd.DataFrame):
    for column in SCORE_COLUMNS:
        if column in diagnosis.columns:
            return column
    return None
 
 
def build_messages(diagnosis: pd.DataFrame, injected_faults=None, config=None):
    table = diagnosis.head(5).to_csv(index=False)
 
    lines = []
    if config:
        lines.append(
            "Experiment settings: " + ", ".join(f"{k}={v}" for k, v in config.items())
        )
    if injected_faults:
        lines.append(
            "Ground truth (only known because this is a simulation): "
            f"faulty robots = {', '.join(injected_faults)}"
        )
    lines.append("Diagnostic ranking (most suspicious first):")
    lines.append(table)
 
    system = (
        "You are a diagnostics assistant for a warehouse robot fleet. "
        "Use ONLY the data given. Never invent robots, numbers or causes. "
        "Write for a non-expert. Use this format: "
        "1) Summary (1-2 sentences) 2) Evidence 3) Recommended action "
        "4) Confidence note (say the ranking is statistical, not proof). "
        "Maximum 180 words."
    )
 
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]
 
 
def fallback_report(diagnosis: pd.DataFrame, injected_faults=None) -> str:
    """Simple offline summary used when the AI service is unavailable."""
    if diagnosis is None or diagnosis.empty or "robot_id" not in diagnosis.columns:
        return "No diagnostic ranking is available to summarise."
 
    score_col = _score_column(diagnosis)
    top = diagnosis.iloc[0]
    text = f"Most suspicious robot: {top['robot_id']}"
    if score_col:
        text += f" (score {float(top[score_col]):.4f})"
    text += "."
 
    if len(diagnosis) > 1:
        others = ", ".join(str(r) for r in diagnosis["robot_id"].iloc[1:3])
        text += f" Next in the ranking: {others}."
 
    if injected_faults:
        if top["robot_id"] in injected_faults:
            text += " This matches an injected fault."
        else:
            text += f" Injected faults were: {', '.join(injected_faults)}."
 
    text += " Recommended action: isolate the top-ranked robot and reassign its failed tasks."
    return text
 
 
def generate_report(
    diagnosis: pd.DataFrame,
    injected_faults=None,
    config=None,
    api_key=None,
    models=None,
    timeout=30,
) -> dict:
    """
    Returns:
      {"text": str,
       "source": "openrouter" | "fallback",
       "model": model that answered (or None),
       "error": readable reason if it fell back (or None),
       "attempts": list of what happened to each model}
    """
    key = get_api_key(api_key)
    attempts = []
 
    if not key:
        return {
            "text": fallback_report(diagnosis, injected_faults),
            "source": "fallback",
            "model": None,
            "error": "No API key found (add OPENROUTER_API_KEY to secrets).",
            "attempts": attempts,
        }
 
    messages = build_messages(diagnosis, injected_faults, config)
 
    for model in get_models(models):
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "X-Title": "FaultTrace AI",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 400,
                    "temperature": 0.2,
                },
                timeout=timeout,
            )
 
            if response.status_code == 401:
                attempts.append(f"{model}: HTTP 401 - the API key is invalid")
                break  # a bad key will fail for every model, stop here
 
            if response.status_code != 200:
                attempts.append(
                    f"{model}: HTTP {response.status_code} - {response.text[:150]}"
                )
                continue
 
            text = response.json()["choices"][0]["message"]["content"].strip()
            if not text:
                attempts.append(f"{model}: empty answer")
                continue
 
            return {
                "text": text,
                "source": "openrouter",
                "model": model,
                "error": None,
                "attempts": attempts,
            }
 
        except Exception as error:
            attempts.append(f"{model}: {error}")
 
    return {
        "text": fallback_report(diagnosis, injected_faults),
        "source": "fallback",
        "model": None,
        "error": " | ".join(attempts) if attempts else "Unknown error.",
        "attempts": attempts,
    }
 