"""L'agent de CreditLens (phase D) : un LLM répond aux questions à partir des outils, puis un garde-fou vérifie.

Deux modes.
- « agent » : le LLM choisit lui-même les outils (il faut un modèle qui sait appeler des outils de façon fiable) ;
- « guidé » : l'application choisit les outils d'après la question (règles simples, `route`), les exécute, et le LLM
  ne fait que rédiger à partir des résultats. Né du premier essai réel : qwen3:1.7b écrivait « je vais appeler
  l'outil motifs_client » sans jamais l'appeler. Un petit modèle local rédige bien, mais choisit mal ses outils.

Déroulé d'une question en mode agent :
  1. le LLM reçoit la question, les règles (SYSTEM) et la liste des outils ;
  2. il appelle des outils ; on les exécute et on lui renvoie les résultats ; il peut en rappeler d'autres ;
  3. quand il écrit sa réponse : si aucun outil n'a été appelé, on le relance UNE fois pour qu'il en appelle ;
  4. le garde-fou (verify.py) contrôle les nombres ; s'il en trouve d'inventés, on demande UNE correction ;
  5. on renvoie la réponse, le statut, la vérification et la trace complète des appels d'outils.
Statuts : « vérifié », « corrigé » (bon après une correction), « non vérifié » (des nombres restent sans source),
« sans outil » (aucun outil appelé), « sans réponse » (texte vide), « interrompu » (trop d'étapes).
"""
import json
import re
import time

from . import agent_tools as T
from . import verify as V

SYSTEM = """Tu es l'assistant de CreditLens, une application de risque de crédit et de détection de fraude.
Tu réponds en français simple, en 3 à 6 phrases, à partir des résultats des outils.

Règles :
1. Pour tout fait ou chiffre, appelle d'abord l'outil adapté. N'écris que des nombres présents dans les résultats des outils (tu peux arrondir à l'unité). Ne fais aucun calcul.
2. Cite les motifs et les conseils renvoyés par les outils, sans en inventer.
3. L'âge, le sexe et la situation maritale ne sont jamais des motifs de refus. Si on te pose la question, dis-le et parle de l'audit d'équité.
4. Un conseil ne garantit rien : rappelle l'avertissement de l'outil.
5. Fraude : ne donne jamais de conseil pour éviter une alerte.
6. Si les outils ne donnent pas l'information, dis simplement que tu ne sais pas.
/no_think"""

NUDGE = ("Tu n'as appelé aucun outil (écrire « je vais appeler l'outil » ne suffit pas). Appelle vraiment les "
         "outils nécessaires, puis réponds uniquement avec ce qu'ils renvoient.")

GUIDED_SYSTEM = """Tu es l'assistant de CreditLens, une application de risque de crédit et de détection de fraude.
Tu réponds en français simple, en 3 à 6 phrases.

Les résultats des outils de CreditLens sont donnés avec la question : c'est ta seule source d'information.
Règles :
1. N'écris que des nombres présents dans ces résultats (tu peux arrondir à l'unité). Ne fais aucun calcul.
2. Cite les motifs et les conseils des résultats, sans en inventer.
3. L'âge, le sexe et la situation maritale ne sont jamais des motifs de refus. Si on te pose la question, dis-le et parle de l'audit d'équité.
4. Un conseil ne garantit rien : rappelle l'avertissement.
5. Fraude : ne donne jamais de conseil pour éviter une alerte.
6. Si les résultats ne donnent pas l'information, dis simplement que tu ne sais pas.
/no_think"""


def correction(unsupported: list[str]) -> str:
    return ("Ta réponse contient des nombres qui ne viennent d'aucun résultat d'outil : " + ", ".join(unsupported)
            + ". Réécris ta réponse en n'utilisant que des nombres présents dans les résultats des outils "
              "(ou supprime ces nombres).")


def ask(llm, ctx: T.ToolContext, question: str, max_steps: int = 8) -> dict:
    t0 = time.time()
    tools = T.schemas()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    allowed = V.collect(question, "question")
    trace, nudges, corrections, answer, check, status = [], 0, 0, "", None, None
    llm_seconds = 0.0
    for _ in range(max_steps):
        resp = llm.chat(messages, tools)
        llm_seconds += resp.get("seconds") or 0.0
        if resp["tool_calls"]:
            messages.append({"role": "assistant", "content": resp["content"] or "",
                             "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in resp["tool_calls"]]})
            for c in resp["tool_calls"]:
                result = T.run_tool(ctx, c["name"], c["arguments"])
                trace.append({"tool": c["name"], "arguments": c["arguments"], "result": result})
                if "erreur" not in result:
                    V.collect(result, c["name"], allowed)
                messages.append({"role": "tool", "tool_name": c["name"],
                                 "content": json.dumps(result, ensure_ascii=False)})
            continue
        answer = resp["content"]
        if not trace and nudges == 0:                        # répond « de mémoire » : une relance
            nudges += 1
            messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": NUDGE}]
            continue
        check = V.verify(answer, allowed)
        if check["unsupported"] and corrections == 0:        # des nombres sans source : une correction
            corrections += 1
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": correction(check["unsupported"])}]
            continue
        if not trace:
            status = "sans outil"
        elif not answer.strip():
            status = "sans réponse"
        elif check["unsupported"]:
            status = "non vérifié"
        else:
            status = "corrigé" if corrections else "vérifié"
        break
    else:
        status = "interrompu"
        check = V.verify(answer, allowed)
    return {"question": question, "answer": answer, "status": status, "verification": check, "trace": trace,
            "mode": "agent", "nudges": nudges, "corrections": corrections,
            "llm": {"provider": llm.provider, "model": llm.model},
            "seconds": round(time.time() - t0, 1), "llm_seconds": round(llm_seconds, 1)}


# ------------------------------------------------------------------------------------------- mode guidé
_CLIENT = re.compile(r"(?:n\s*°|\bno\.?|num[ée]ro|client|transaction|dossier)\s*:?\s*(?:n\s*°\s*)?(\d+)", re.I)


def _norm(text: str) -> str:
    return V._norm(text)


def route(question: str, ctx: T.ToolContext) -> list[tuple[str, dict]]:
    """Choisit les outils d'après des mots de la question. Volontairement simple et lisible (pas de LLM ici)."""
    q = _norm(question)
    calls = []
    m = _CLIENT.search(question)
    if m:                                                    # un client précis : décision, motifs, conseil
        n = int(m.group(1))
        calls += [("decision_client", {"numero": n}), ("motifs_client", {"numero": n}), ("conseil_client", {"numero": n})]
    if re.search(r"equit|femme|homme|sexe|\bage\b|ages\b|jeune|vieu|groupe|discrimin|audit|acces au conseil|marie|celibat", q):
        calls.append(("audit_recours", {}))
    if "fraud" in q:
        calls.append(("fraude_resume", {}))
    if re.search(r"modele|seuil|cout|evaluation|performance|\bauc\b|fiab|lot de test", q):
        calls.append(("infos_modele", {}))
    for term in T.GLOSSARY:
        if _norm(term) in q:
            calls.append(("definir", {"terme": term}))
            break
    if not calls:                                            # rien de reconnu : au moins le contexte général
        calls.append(("infos_modele", {}))
    seen, out = set(), []
    for name, args in calls:
        key = (name, json.dumps(args, sort_keys=True))
        if key not in seen:
            seen.add(key)
            out.append((name, args))
    return out


def ask_guided(llm, ctx: T.ToolContext, question: str) -> dict:
    t0 = time.time()
    allowed = V.collect(question, "question")
    trace = []
    for name, args in route(question, ctx):
        result = T.run_tool(ctx, name, args)
        trace.append({"tool": name, "arguments": args, "result": result, "by": "application"})
        if "erreur" not in result:
            V.collect(result, name, allowed)
    facts = "\n".join(f"- {t['tool']} : {json.dumps(t['result'], ensure_ascii=False)}" for t in trace)
    messages = [{"role": "system", "content": GUIDED_SYSTEM},
                {"role": "user", "content": f"{question}\n\nRésultats des outils de CreditLens (ta seule source) :\n{facts}"}]
    corrections, llm_seconds = 0, 0.0
    while True:
        resp = llm.chat(messages, None)                      # aucun outil proposé : le LLM ne fait que rédiger
        llm_seconds += resp.get("seconds") or 0.0
        answer = resp["content"]
        check = V.verify(answer, allowed)
        if check["unsupported"] and corrections == 0 and answer.strip():
            corrections += 1
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": correction(check["unsupported"])}]
            continue
        break
    if not answer.strip():
        status = "sans réponse"
    elif check["unsupported"]:
        status = "non vérifié"
    else:
        status = "corrigé" if corrections else "vérifié"
    return {"question": question, "answer": answer, "status": status, "verification": check, "trace": trace,
            "mode": "guidé", "nudges": 0, "corrections": corrections,
            "llm": {"provider": llm.provider, "model": llm.model},
            "seconds": round(time.time() - t0, 1), "llm_seconds": round(llm_seconds, 1)}
