"""Agent (phase D) : outils, garde-fou des nombres, boucle, client Ollama, routes.

Aucun vrai LLM ici : un faux LLM scripté (llm.ScriptedClient) joue les réponses, et un faux serveur HTTP
(httpx.MockTransport) joue Ollama. Les essais avec le vrai modèle se font sur un PC où tourne Ollama.
"""
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import agent as AG
from app import agent_tools as T
from app import llm as LLM
from app import models as MD
from app import prep as P
from app import verify as V
from app.main import app, get_llm_factory
from app.state import Session
from test_recourse_future import _monthly

THR = 0.25


@pytest.fixture(scope="module")
def ctx():
    s = Session()
    s.df, s.target, s.positive, s.name = _monthly(), "default", "1", "taiwan_credit"
    P.build(s, {"features": [c for c in s.df.columns if c != "default"], "test_size": 0.25, "random_state": 0,
                "num_impute": "median", "cat_impute": "mode", "scaling": "standard", "stratify": True})
    est, info = MD.train_one("xgb", {}, False, False, s.Z_train, s.y_train.values)
    s.models["xgb"] = {"est": est, "info": info, "label": "XGBoost"}
    return T.ToolContext(s, "xgb", THR)


def _refused(ctx):
    p = ctx.sess.models["xgb"]["est"].predict_proba(ctx.sess.Z_test)[:, 1]
    return int(np.where(p >= THR)[0][0])


def _last_tool(messages):
    return json.loads(next(m for m in reversed(messages) if m["role"] == "tool")["content"])


# ----------------------------------------------------------------------------------------------- garde-fou
def test_numbers_are_read_like_a_french_reader_would():
    vals = {t["text"]: (t["values"], t["decimals"]) for t in V.numbers(
        "Risque 34,8 % ; 0.348 ; 1 859 par mois ; au moins 11 000 ; 11,000 ; client n°12.")}
    assert vals["34,8 %"] == ([34.8], 1) and vals["0.348"] == ([0.348], 3)
    assert vals["1 859"] == ([1859.0], 0) and vals["11 000"] == ([11000.0], 0)
    assert vals["11,000"][0] == [11.0, 11000.0]                  # ambigu : les deux lectures sont essayées
    assert vals["12"] == ([12.0], 0)
    assert V.numbers("1. Premier motif\n2) Second motif") == []    # les numéros de liste ne comptent pas


def test_a_number_is_supported_only_up_to_the_displayed_rounding():
    allowed = [(34.8, "decision_client"), (1859.33, "conseil_client"), (12.0, "question")]
    r = V.verify("Risque 35 %, ou plus précisément 34,8 %, pour le client 12, qui rembourse 1 859.", allowed)
    assert r["unsupported"] == [] and r["n_supported"] == 4
    assert {f["source"] for f in r["numbers"]} == {"decision_client", "conseil_client", "question"}
    bad = V.verify("Risque 34 % (ou 34,9 %), il rembourse 1 860.", allowed)
    assert bad["unsupported"] == ["34 %", "34,9 %", "1 860"]


def test_no_conversion_is_allowed():
    assert V.verify("Le coût est de 54,8 %.", [(0.548, "infos_modele")])["unsupported"] == ["54,8 %"]


def test_warnings_for_sensitive_words_and_promises():
    assert V.verify("Le refus vient de son âge.", [])["warnings"]
    assert V.verify("Vous serez accepté si vous payez.", [])["warnings"]
    assert V.verify("Ce genre de client paie à temps.", [])["warnings"] == []
    assert V.verify("Vous seriez accepté si le risque passait sous le seuil.", [])["warnings"] == []   # hypothèse
    for ok in ("Ce conseil ne garantit rien.", "Il ne garantit ni le comportement futur ni la décision.",
               "Ce n'est pas garanti.", "Il n'y a aucune garantie."):
        assert V.verify(ok, [])["warnings"] == [], ok                   # l'avertissement attendu n'est pas une promesse
    assert V.verify("C'est garanti : il passera.", [])["warnings"]


def test_collect_reads_numbers_inside_tool_texts():
    got = V.collect({"a": 0.5, "b": [{"texte": "risque de 15.4% au lieu de 34.8%"}], "ok": True}, "outil")
    assert sorted(v for v, _ in got) == [0.5, 15.4, 34.8]          # le booléen n'est pas un nombre


# ----------------------------------------------------------------------------------------------- outils
def test_decision_tool_matches_the_model(ctx):
    i = _refused(ctx)
    d = T.run_tool(ctx, "decision_client", {"numero": i})
    p = float(ctx.sess.models["xgb"]["est"].predict_proba(ctx.sess.Z_test[[i]])[0, 1])
    assert d["risque_annonce_pct"] == round(100 * p, 1) and d["decision"] == "refus"
    assert d["seuil_pct"] == 25.0 and d["ecart_au_seuil_points"] == round(100 * (p - THR), 1)


def test_tool_errors_are_readable_messages_never_exceptions(ctx):
    assert "n'existe pas" in T.run_tool(ctx, "decision_client", {"numero": 10 ** 6})["erreur"]
    assert "invalide" in T.run_tool(ctx, "decision_client", {"numero": "douze"})["erreur"]
    assert "arguments attendus" in T.run_tool(ctx, "motifs_client", {})["erreur"]
    assert "outil inconnu" in T.run_tool(ctx, "supprimer_client", {"numero": 1})["erreur"]


def test_reason_and_advice_tools(ctx):
    i = _refused(ctx)
    m = T.run_tool(ctx, "motifs_client", {"numero": i})
    assert m["motifs_vers_le_refus"] and not {"AGE", "SEX"} & {x["texte"].split(" :")[0] for x in m["motifs_vers_le_refus"]}
    assert set(m["variables_sensibles_jamais_citees"]) == {"AGE", "SEX"}
    c = T.run_tool(ctx, "conseil_client", {"numero": i})
    assert c["statut"] in ("found", "none_found") and "ne garantit" in c["avertissement"]
    if c["statut"] == "found":
        assert all(o["delai_mois"] in (1, 3, 6) and o["nouveau_risque_pct"] < 25 for o in c["conseils"])


def test_audit_tool_is_cached(ctx, monkeypatch):
    calls = []
    real = T.RA.audit
    monkeypatch.setattr(T.RA, "audit", lambda *a, **k: calls.append(1) or real(*a, **k))
    a = T.run_tool(ctx, "audit_recours", {})
    b = T.run_tool(ctx, "audit_recours", {})
    assert a == b and len(calls) == 1 and a["statut"] == "ok"


def test_model_info_fraud_summary_and_definitions(ctx):
    info = T.run_tool(ctx, "infos_modele", {})
    assert info["modele_explique"] == "XGBoost" and "pas encore faite" in info["evaluation_finale"]
    assert "pas encore calculé" in T.run_tool(ctx, "fraude_resume", {})["statut"]
    assert "16,7 %" in T.run_tool(ctx, "definir", {"terme": "Seuil"})["definition"]
    assert "termes_connus" in T.run_tool(ctx, "definir", {"terme": "blockchain"})


def test_schemas_describe_every_tool():
    sch = T.schemas()
    assert {x["function"]["name"] for x in sch} == set(T.TOOLS)
    assert all(x["type"] == "function" and x["function"]["parameters"]["type"] == "object" for x in sch)


# ----------------------------------------------------------------------------------------------- boucle
def _honest(messages):
    d = _last_tool(messages)
    return f"Le client n°{d['numero']} a un risque annoncé de {d['risque_annonce_pct']} %, pour un seuil de {d['seuil_pct']} % : {d['decision']}."


def test_happy_path_is_verified_and_traced(ctx):
    i = _refused(ctx)
    llm = LLM.ScriptedClient([[("decision_client", {"numero": i})], _honest])
    r = AG.ask(llm, ctx, f"Pourquoi le client n°{i} est-il refusé ?")
    assert r["status"] == "vérifié" and r["verification"]["unsupported"] == []
    assert [t["tool"] for t in r["trace"]] == ["decision_client"]
    sent = llm.received[1]                                           # ce que le LLM a reçu au 2e tour
    assert sent[2]["role"] == "assistant" and sent[2]["tool_calls"][0]["function"]["name"] == "decision_client"
    assert sent[3]["role"] == "tool" and sent[3]["tool_name"] == "decision_client"


def test_an_invented_number_triggers_one_correction(ctx):
    i = _refused(ctx)
    llm = LLM.ScriptedClient([[("decision_client", {"numero": i})], "Le risque est de 42,0 %.", _honest])
    r = AG.ask(llm, ctx, f"Risque du client {i} ?")
    assert r["status"] == "corrigé" and r["corrections"] == 1
    assert "42,0 %" in llm.received[2][-1]["content"]                # la demande de correction cite le nombre


def test_a_persistent_lie_is_reported_not_hidden(ctx):
    i = _refused(ctx)
    llm = LLM.ScriptedClient([[("decision_client", {"numero": i})], "Le risque est de 42,0 %.", "Disons 43 %."])
    r = AG.ask(llm, ctx, f"Risque du client {i} ?")
    assert r["status"] == "non vérifié" and r["verification"]["unsupported"] == ["43 %"]


def test_answering_from_memory_gets_one_nudge(ctx):
    i = _refused(ctx)
    ok = AG.ask(LLM.ScriptedClient(["Il est refusé.", [("decision_client", {"numero": i})], _honest]), ctx, "Client 3 ?")
    assert ok["status"] == "vérifié" and ok["nudges"] == 1
    never = AG.ask(LLM.ScriptedClient(["Il est refusé.", "Il est vraiment refusé."]), ctx, "Client 3 ?")
    assert never["status"] == "sans outil" and never["trace"] == []


def test_endless_tool_calls_are_interrupted(ctx):
    llm = LLM.ScriptedClient([[("infos_modele", {})]] * 20)
    r = AG.ask(llm, ctx, "Parle-moi du modèle.", max_steps=4)
    assert r["status"] == "interrompu" and len(r["trace"]) == 4


def test_numbers_from_tool_errors_are_not_facts(ctx):
    llm = LLM.ScriptedClient([[("decision_client", {"numero": 10 ** 6})], "Les numéros vont de 0 à 499.",
                              "Ce client n'existe pas dans le lot de test."])
    r = AG.ask(llm, ctx, "Client 1000000 ?")
    assert "erreur" in r["trace"][0]["result"]
    assert "499" in llm.received[2][-1]["content"]                   # demande de correction : 499 n'a pas de source
    assert r["status"] == "corrigé"


def test_an_empty_answer_is_never_called_verified(ctx):
    llm = LLM.ScriptedClient([[("infos_modele", {})], "Le seuil est de 99 %."])   # puis plus rien : texte vide
    r = AG.ask(llm, ctx, "Quel seuil ?")
    assert r["status"] == "sans réponse" and r["corrections"] == 1


# ----------------------------------------------------------------------------------------------- Ollama
def _ollama(handler):
    return LLM.OllamaClient(model="qwen3:8b", url="http://ollama.test", transport=httpx.MockTransport(handler))


def test_ollama_request_and_tool_call_parsing():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "decision_client", "arguments": {"numero": 12}}},
            {"function": {"name": "motifs_client", "arguments": "{\"numero\": 12}"}}]}, "eval_count": 30})
    r = _ollama(handler).chat([{"role": "user", "content": "x"}], T.schemas())
    assert seen["model"] == "qwen3:8b" and seen["stream"] is False and seen["think"] is False
    assert seen["options"] == {"temperature": 0, "num_ctx": 8192} and len(seen["tools"]) == len(T.TOOLS)
    assert r["tool_calls"] == [{"name": "decision_client", "arguments": {"numero": 12}},
                               {"name": "motifs_client", "arguments": {"numero": 12}}]


def test_ollama_thinking_is_removed_and_old_versions_are_handled():
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "invalid option: think"})
        return httpx.Response(200, json={"message": {"content": "<think>je réfléchis…</think>\n\nRéponse finale."}})
    r = _ollama(handler).chat([{"role": "user", "content": "x"}])
    assert r["content"] == "Réponse finale." and "think" in bodies[0] and "think" not in bodies[1]


def test_ollama_tool_calls_written_as_text_are_recovered():
    def handler(req):
        return httpx.Response(200, json={"message": {"content": 'Je vérifie.<tool_call>{"name": "decision_client", '
                                                                  '"arguments": {"numero": 7}}</tool_call>'}})
    r = _ollama(handler).chat([{"role": "user", "content": "x"}])
    assert r["tool_calls"] == [{"name": "decision_client", "arguments": {"numero": 7}}] and r["content"] == "Je vérifie."


def test_ollama_errors_are_explained():
    def missing(req):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen3:4b"}]})
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})
    with pytest.raises(LLM.LLMError, match="qwen3:4b"):
        _ollama(missing).chat([{"role": "user", "content": "x"}])

    def down(req):
        raise httpx.ConnectError("refusé")
    with pytest.raises(LLM.LLMError, match="Ollama ne répond pas"):
        _ollama(down).chat([{"role": "user", "content": "x"}])


def test_ollama_picks_the_largest_chat_model():
    """Les modèles installés chez Emna le 5 novembre, plus qwen3:4b : c'est lui qui doit être choisi par défaut."""
    def tags(req):
        return httpx.Response(200, json={"models": [
            {"name": "qwen3:1.7b", "size": 1.4e9, "details": {"parameter_size": "2.0B"}},
            {"name": "nomic-embed-text:latest", "size": 2.7e8, "details": {"parameter_size": "137M"}},
            {"name": "qwen3.5:0.8b", "size": 1.0e9, "details": {"parameter_size": "873.44M"}},
            {"name": "qwen3.5:2b", "size": 2.7e9, "details": {"parameter_size": "2.3B"}},   # fichier plus gros…
            {"name": "qwen3:4b", "size": 2.5e9, "details": {"parameter_size": "4.0B"}}]})  # … mais modèle plus grand
    c = LLM.OllamaClient(url="http://ollama.test", transport=httpx.MockTransport(tags))
    assert c.model == "qwen3:4b"
    assert LLM.param_count("752.39M") == pytest.approx(7.5239e8) and LLM.param_count(None) == 0
    assert LLM.pick_default([{"name": "nomic-embed-text:latest", "parameters": "137M"}]) is None


# ----------------------------------------------------------------------------------------------- routes
def test_agent_routes():
    from test_reasons import _credit
    import io
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-agent"}
    buf = io.StringIO()
    _credit().to_csv(buf, index=False)
    c.post("/api/data/upload", files={"file": ("c.csv", buf.getvalue().encode(), "text/csv")}, data={"target": "y"}, headers=H)
    c.post("/api/prep", json={"features": ["age", "sex", "debt", "paid", "job"]}, headers=H)
    c.post("/api/models/train", json={"models": [{"name": "xgb"}]}, headers=H)

    def answer(messages):
        d = _last_tool(messages)
        return f"Risque de {d['risque_annonce_pct']} % : {d['decision']}."
    app.dependency_overrides[get_llm_factory] = lambda: (lambda name=None: LLM.ScriptedClient([[("decision_client", {"numero": 0})], answer]))
    try:
        r = c.post("/api/agent/ask", json={"question": "Et le client 0 ?", "model": "xgb", "threshold": 0.3,
                                            "mode": "agent"}, headers=H)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "vérifié" and r.json()["llm"]["provider"] == "scripted"
        assert r.json()["mode"] == "agent"
        app.dependency_overrides[get_llm_factory] = lambda: (lambda name=None: LLM.ScriptedClient(["Je ne sais pas."]))
        g = c.post("/api/agent/ask", json={"question": "Et le client 0 ?", "model": "xgb", "threshold": 0.3}, headers=H)
        assert g.status_code == 200 and g.json()["mode"] == "guidé"                       # mode par défaut
        assert c.post("/api/agent/ask", json={"question": "Et le client 0 ?", "model": "xgb", "mode": "magique"},
                      headers=H).status_code == 422
        assert c.post("/api/agent/ask", json={"question": "Et le client 0 ?", "model": "rf"}, headers=H).status_code == 404

        def boom(name=None):
            raise LLM.LLMError("Ollama ne répond pas")
        app.dependency_overrides[get_llm_factory] = lambda: boom
        r = c.post("/api/agent/ask", json={"question": "Et le client 0 ?", "model": "xgb"}, headers=H)
        assert r.status_code == 503 and "Ollama" in r.json()["detail"]
        assert c.get("/api/agent/status").json()["ok"] is False

        tags = httpx.MockTransport(lambda req: httpx.Response(200, json={"models": [{"name": "qwen3:8b", "size": 5.2e9}]}))
        app.dependency_overrides[get_llm_factory] = lambda: (lambda name=None: LLM.OllamaClient(model=name, url="http://o.test", transport=tags))
        st = c.get("/api/agent/status").json()
        assert st["ok"] and st["default"] == "qwen3:8b" and st["models"][0]["size_gb"] == 5.2
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------------------------------- mode guidé
@pytest.mark.parametrize("question, expected", [
    ("Pourquoi le client n°12 est-il refusé ?", ["decision_client", "motifs_client", "conseil_client"]),
    ("Que peut faire le client 12 pour être accepté ?", ["decision_client", "motifs_client", "conseil_client"]),
    ("Le client n°12 est-il refusé à cause de son âge ?",
     ["decision_client", "motifs_client", "conseil_client", "audit_recours"]),
    ("L'accès au conseil est-il le même pour les femmes et les hommes ?", ["audit_recours"]),
    ("Qu'est-ce que le seuil de décision ?", ["infos_modele", "definir"]),
    ("Combien de pertes la détection de fraude évite-t-elle ?", ["fraude_resume"]),
    ("Bonjour !", ["infos_modele"]),
])
def test_routing_picks_the_tools_from_the_question(ctx, question, expected):
    calls = AG.route(question, ctx)
    assert [n for n, _ in calls] == expected
    assert all(a == {"numero": 12} for n, a in calls if n.endswith("_client"))


def test_guided_mode_only_asks_the_llm_to_write(ctx):
    i = _refused(ctx)

    def write(messages):
        facts = messages[-1]["content"]
        assert "decision_client" in facts and "Résultats des outils" in facts
        d = json.loads(facts.split("- decision_client : ")[1].split("\n")[0])
        return f"Le client n°{i} est en {d['decision']} : risque annoncé {d['risque_annonce_pct']} %, seuil {d['seuil_pct']} %."
    llm = LLM.ScriptedClient([write])
    r = AG.ask_guided(llm, ctx, f"Pourquoi le client n°{i} est-il refusé ?")
    assert r["status"] == "vérifié" and r["mode"] == "guidé"
    assert llm.tools_received == [None]                              # aucun outil proposé au LLM
    assert [t["tool"] for t in r["trace"]] == ["decision_client", "motifs_client", "conseil_client"]
    assert all(t["by"] == "application" for t in r["trace"])


def test_guided_mode_still_corrects_and_reports(ctx):
    i = _refused(ctx)
    ok = AG.ask_guided(LLM.ScriptedClient(["Risque de 42,0 %.", "Le client est refusé."]), ctx, f"Client n°{i} ?")
    assert ok["status"] == "corrigé" and ok["corrections"] == 1
    bad = AG.ask_guided(LLM.ScriptedClient(["Risque de 42,0 %.", "Risque de 43 %."]), ctx, f"Client n°{i} ?")
    assert bad["status"] == "non vérifié" and bad["verification"]["unsupported"] == ["43 %"]
    empty = AG.ask_guided(LLM.ScriptedClient([""]), ctx, f"Client n°{i} ?")
    assert empty["status"] == "sans réponse"
