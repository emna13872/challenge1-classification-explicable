"""L'agent de CreditLens (phase D) : un LLM répond aux questions en appelant les outils, puis un garde-fou vérifie.

Déroulé d'une question :
  1. le LLM reçoit la question, les règles (SYSTEM) et la liste des outils ;
  2. il appelle des outils ; on les exécute et on lui renvoie les résultats ; il peut en rappeler d'autres ;
  3. quand il écrit sa réponse : si aucun outil n'a été appelé, on le relance UNE fois pour qu'il en appelle ;
  4. le garde-fou (verify.py) contrôle les nombres ; s'il en trouve d'inventés, on demande UNE correction ;
  5. on renvoie la réponse, le statut, la vérification et la trace complète des appels d'outils.
Statuts : « vérifié », « corrigé » (bon après une correction), « non vérifié » (des nombres restent sans source),
« sans outil » (aucun outil appelé), « sans réponse » (texte vide), « interrompu » (trop d'étapes).
"""
import json
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

NUDGE = ("Tu n'as appelé aucun outil. Appelle d'abord les outils nécessaires, puis réponds uniquement avec "
         "ce qu'ils renvoient.")


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
            "nudges": nudges, "corrections": corrections, "llm": {"provider": llm.provider, "model": llm.model},
            "seconds": round(time.time() - t0, 1), "llm_seconds": round(llm_seconds, 1)}
