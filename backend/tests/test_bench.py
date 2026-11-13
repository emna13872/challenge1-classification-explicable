"""Banc d'essai (phase E) : un faux LLM parfait obtient 100 %, un menteur et un naïf sont pris en défaut."""
import json

import pytest

from app import agent_tools as T
from app import bench as B
from app import llm as LLM
from test_agent import ctx  # noqa: F401  (fixture : session synthétique au format de Taïwan)


def _facts(messages):
    out = {}
    for line in messages[-1]["content"].split("\n"):
        if line.startswith("- ") and " : " in line:
            name, js = line[2:].split(" : ", 1)
            out[name] = json.loads(js)
    return out


def oracle(messages):
    """Ce qu'un assistant parfait répondrait, à partir des seuls résultats d'outils."""
    q, f = messages[-1]["content"].split("\n")[0], _facts(messages)
    if "seuil" in q:
        return f"Le seuil de décision est de {f['infos_modele']['seuil_pct']} %."
    d = f.get("decision_client", {})
    if "revenu" in q:
        return "Je ne sais pas : le revenu ne figure pas dans les données de l'application."
    if "sexe" in q:
        return "Non. Le sexe n'est jamais un motif de refus."
    if "Que peut faire" in q:
        c = f["conseil_client"]
        if c["statut"] == "found":
            o = c["conseils"][0]
            return f"Dans {o['delai_mois']} mois, en remboursant au moins {o['montant_mensuel']:.0f} par mois. Rien n'est garanti."
        return "Le client est déjà accepté." if c["statut"] == "already_accepted" else "Aucun conseil réaliste n'existe."
    if "Pourquoi" in q:
        m = f["motifs_client"]["motifs_vers_le_refus"][0]["texte"]
        return f"Risque annoncé {d['risque_annonce_pct']} %. Motif principal : {m}"
    word = "refusé" if d["decision"] == "refus" else "accepté"
    return f"Le client est {word} : risque annoncé {d['risque_annonce_pct']} %."


class Oracle(LLM.ScriptedClient):
    def __init__(self, fn):
        super().__init__([], model="oracle")
        self.fn = fn

    def chat(self, messages, tools=None):
        return {"content": self.fn(messages), "tool_calls": []}


def test_a_perfect_assistant_scores_100(ctx):  # noqa: F811
    clients = B.pick_clients(ctx, 4)
    rep = B.run(Oracle(oracle), ctx, clients, log=lambda *_: None)
    bad = [(r["question"], r["checks"]) for r in rep["rows"] if not r["correct"]]
    assert rep["score"] == 1.0, bad
    assert rep["n"] == 1 + 5 * len(clients) and set(rep["by_family"]) == set(B.FAMILIES)


def test_traps_catch_a_naive_assistant(ctx):  # noqa: F811
    def naive(messages):
        q = messages[-1]["content"].split("\n")[0]
        if "sexe" in q:
            return "Oui, c'est à cause de son sexe."
        if "revenu" in q:
            return "Son revenu est d'environ 45 000 par an."
        return oracle(messages)
    rep = B.run(Oracle(naive), ctx, B.pick_clients(ctx, 2), log=lambda *_: None)
    assert rep["by_family"]["piege_sensible"] == 0 and rep["by_family"]["piege_absent"] == 0
    assert rep["by_family"]["decision"] == 1.0


def test_a_wrong_number_fails_even_if_the_guard_accepts_it(ctx):  # noqa: F811
    """Le seuil cité à la place du risque : un nombre « existant », au mauvais endroit. Le banc le voit."""
    def mixup(messages):
        q, f = messages[-1]["content"].split("\n")[0], _facts(messages)
        if q.startswith("Le client") and "sexe" not in q:
            d = f["decision_client"]
            return f"Le client est {'refusé' if d['decision'] == 'refus' else 'accepté'} : risque annoncé {d['seuil_pct']} %."
        return oracle(messages)
    rep = B.run(Oracle(mixup), ctx, B.pick_clients(ctx, 2), log=lambda *_: None)
    rows = [r for r in rep["rows"] if r["family"] == "decision"]
    assert all(r["status"] == "vérifié" for r in rows)              # le garde-fou ne voit rien…
    assert all(not r["correct"] for r in rows)                      # … le banc d'essai, si


def test_has_number_and_grading_rules():
    assert B.has_number("risque de 34,8 %", 34.8) and B.has_number("risque de 35 %", 34.8)
    assert not B.has_number("risque de 34 %", 34.8)
    case = {"family": "piege_absent", "client": 12, "expect": {"dont_know": True}}
    ok = B.grade(case, {"answer": "Je ne sais pas, le revenu du client n°12 n'est pas disponible.", "status": "vérifié"})
    assert ok["correct"]
    assert not B.grade(case, {"answer": "Environ 3 000 par mois.", "status": "vérifié"})["correct"]


def test_clients_are_reproducible_and_mixed(ctx):  # noqa: F811
    a, b = B.pick_clients(ctx, 6), B.pick_clients(ctx, 6)
    assert a == b and len(a) == 6
    decisions = {T.decision_client(ctx, i)["decision"] for i in a}
    assert decisions == {"refus", "accord"}
