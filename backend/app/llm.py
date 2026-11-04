"""Les LLM que l'agent sait utiliser : Ollama (local, gratuit) et un faux LLM scripté pour les tests.

Interface commune : chat(messages, tools) -> {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}.
L'agent ne connaît que cette interface : ajouter un autre fournisseur (Claude, OpenAI…) = une classe de plus.
"""
import json
import os
import re
import time

import httpx

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("CREDITLENS_LLM")          # sinon : le plus grand modèle de conversation installé
_THINK = re.compile(r"<think>.*?</think>", re.S)


def param_count(text: str | None) -> float:
    """« 4.0B » -> 4e9, « 752.39M » -> 7.5e8 (taille annoncée par Ollama) ; 0 si inconnue."""
    m = re.match(r"\s*([\d.]+)\s*([KMBT])", text or "", re.I)
    return float(m.group(1)) * {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2).upper()] if m else 0.0


def pick_default(models: list[dict]) -> str | None:
    """Modèle par défaut : le plus grand modèle de conversation installé (les petits appellent mal les outils)."""
    usable = [m for m in models if "embed" not in m["name"]]          # un modèle d'« embedding » ne sait pas écrire
    return max(usable, key=lambda m: param_count(m.get("parameters")))["name"] if usable else None


class LLMError(Exception):
    """Le LLM est injoignable ou a refusé la requête : message en clair pour l'utilisateur."""


def clean(text: str | None) -> str:
    """Retire le « raisonnement » que certains modèles (qwen3) écrivent entre <think> et </think>."""
    t = _THINK.sub("", text or "")
    if "<think>" in t:                                       # balise jamais refermée : on garde ce qui précède
        t = t.split("<think>")[0]
    return t.strip()


_TEXT_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_calls(message: dict) -> list[dict]:
    calls = []
    if not message.get("tool_calls") and "<tool_call>" in (message.get("content") or ""):
        # secours : certains modèles écrivent l'appel d'outil dans le texte au lieu du champ prévu
        for raw in _TEXT_CALL.findall(message["content"]):
            try:
                c = json.loads(raw)
                calls.append({"name": c.get("name", ""), "arguments": c.get("arguments") or {}})
            except json.JSONDecodeError:
                pass
        return calls
    for c in message.get("tool_calls") or []:
        f = c.get("function", c)
        args = f.get("arguments") or {}
        if isinstance(args, str):                            # certains serveurs envoient du JSON en texte
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": f.get("name", ""), "arguments": args if isinstance(args, dict) else {}})
    return calls


class OllamaClient:
    """Ollama (http://localhost:11434) : /api/chat avec outils, température 0, contexte de 8 192 jetons."""

    provider = "ollama"

    def __init__(self, model: str | None = None, url: str = DEFAULT_URL, timeout: float = 600.0,
                 num_ctx: int = 8192, transport=None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._model = model or DEFAULT_MODEL                  # choisi au premier appel si rien n'est imposé
        self.think_supported = True

    @property
    def model(self) -> str:
        if not self._model:
            self._model = self._pick_model()
        return self._model

    def models(self) -> list[dict]:
        try:
            r = self._http.get(f"{self.url}/api/tags", timeout=10)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama ne répond pas sur {self.url} ({type(e).__name__}). Lancez l'application Ollama "
                           "(ou « ollama serve » dans un terminal), puis réessayez.") from e
        return [{"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1),
                 "parameters": (m.get("details") or {}).get("parameter_size")} for m in r.json().get("models", [])]

    def _pick_model(self) -> str:
        name = pick_default(self.models())
        if not name:
            raise LLMError("Aucun modèle Ollama installé. Exemple : « ollama pull qwen3:4b ».")
        return name

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_ctx": self.num_ctx}}
        if tools:
            body["tools"] = tools
        if self.think_supported:
            body["think"] = False                            # qwen3 : pas de longue « réflexion » cachée, plus rapide
        t0 = time.time()
        try:
            r = self._http.post(f"{self.url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama ne répond pas sur {self.url} ({type(e).__name__}). Lancez Ollama, puis réessayez.") from e
        if r.status_code >= 400:
            err = (r.json().get("error") if r.headers.get("content-type", "").startswith("application/json") else r.text) or ""
            if "think" in err.lower() and self.think_supported:      # vieux Ollama ou modèle sans option « think »
                self.think_supported = False
                return self.chat(messages, tools)
            if "not found" in err.lower():
                raise LLMError(f"Modèle « {self.model} » introuvable dans Ollama. Modèles installés : "
                               f"{', '.join(m['name'] for m in self.models()) or 'aucun'}.")
            if "does not support tools" in err.lower():
                raise LLMError(f"Le modèle « {self.model} » ne sait pas appeler d'outils. Choisissez qwen3 (ou llama3.1).")
            raise LLMError(f"Erreur d'Ollama ({r.status_code}) : {err}")
        data = r.json()
        msg = data.get("message") or {}
        calls = _parse_calls(msg)
        content = _TEXT_CALL.sub("", msg.get("content") or "") if calls else msg.get("content")
        return {"content": clean(content), "tool_calls": calls,
                "seconds": round(time.time() - t0, 1), "tokens": data.get("eval_count")}


class ScriptedClient:
    """Faux LLM pour les tests : renvoie, dans l'ordre, des réponses écrites à l'avance (et garde ce qu'on lui a envoyé).

    Chaque étape est soit un texte (réponse finale), soit une liste d'appels d'outils [(nom, arguments), …],
    soit une fonction(messages) -> l'un des deux (pour répondre selon ce que les outils ont renvoyé).
    """

    provider = "scripted"

    def __init__(self, steps, model: str = "faux-llm"):
        self.steps = list(steps)
        self.model = model
        self.received: list[list[dict]] = []
        self.tools_received: list = []

    def chat(self, messages, tools=None):
        self.received.append([dict(m) for m in messages])
        self.tools_received.append(tools)
        if not self.steps:
            return {"content": "", "tool_calls": []}
        step = self.steps.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, str):
            return {"content": clean(step), "tool_calls": []}
        return {"content": "", "tool_calls": [{"name": n, "arguments": a} for n, a in step]}
