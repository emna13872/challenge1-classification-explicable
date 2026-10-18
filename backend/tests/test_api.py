"""Test de bout en bout de l'API sur le jeu Telco (téléchargement réel)."""
import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

H = {"X-Session-Id": "pytest"}
c = TestClient(app)


def ok(r):
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def prepared():
    s = ok(c.post("/api/data/load", json={"dataset": "telco"}, headers=H))
    assert s["target"] == "Churn" and s["n_rows"] > 7000
    feats = [col["name"] for col in s["columns"] if col["name"] != "Churn" and not col["suggest_exclude"]]
    p = ok(c.post("/api/prep", json={"features": feats}, headers=H))
    assert p["n_train"] + p["n_test"] == s["n_rows"] - p["dropped_rows"]
    return p


def test_totalcharges_is_numeric(prepared):
    s = ok(c.get("/api/data/summary", headers=H))
    kinds = {col["name"]: col["kind"] for col in s["columns"]}
    assert kinds["TotalCharges"] == "numeric"          # régression : colonnes texte « str » de pandas 3
    assert prepared["n_features_encoded"] < 80


def test_data_views(prepared):
    ok(c.get("/api/data/correlation", headers=H))
    v = ok(c.get("/api/data/column/tenure", headers=H))
    assert v["kind"] == "numeric" and v["rate"]
    v = ok(c.get("/api/data/column/Contract", headers=H))
    assert v["kind"] == "categorical"


def test_train_evaluate(prepared):
    specs = [{"name": n} for n in ["logreg", "tree", "knn", "rf", "xgb"]]
    t = ok(c.post("/api/models/train", json={"models": specs}, headers=H))
    assert len(t["trained"]) == 5
    ev = ok(c.get("/api/evaluate?threshold=0.4", headers=H))
    for name, m in ev["models"].items():
        assert 0.6 < m["auc"] < 1.0, name
        assert math.isclose(m["gini"], 2 * m["auc"] - 1)
        assert len(m["table"]["rows"]) == 101
    assert ev["models"]["logreg"]["auc"] > 0.80


def test_tuning(prepared):
    t = ok(c.post("/api/models/train", json={"models": [{"name": "tree", "tune": True}]}, headers=H))
    assert "cv_auc" in t["trained"][0]


@pytest.mark.parametrize("name", ["logreg", "tree", "knn", "rf", "xgb"])
def test_explain_global(prepared, name):
    g = ok(c.get(f"/api/explain/global/{name}", headers=H))
    assert g["shap"]["beeswarm"] and g["shap"]["dependence"]
    assert len(g["shap"]["beeswarm"][0]["shap"]) == g["shap"]["n_samples"]


def test_permutation(prepared):
    p = ok(c.get("/api/explain/permutation?models=logreg,xgb", headers=H))
    assert set(p) == {"logreg", "xgb"} and p["xgb"]["items"]


@pytest.mark.parametrize("name", ["logreg", "xgb", "rf"])
def test_local_test_index(prepared, name):
    r = ok(c.post("/api/explain/local", json={"model": name, "test_index": 3, "threshold": 0.4}, headers=H))
    w = r["waterfall"]
    assert r["sentence"].startswith("Le risque")
    assert r["lime"]["items"] and r["comparison"] and "top5_overlap" in r["agreement"]
    # additivité SHAP : base + somme des contributions = sortie du modèle (dans son espace)
    total = w["base"] + sum(i["shap"] for i in w["items"]) + w["others"]
    assert math.isclose(total, w["output"], rel_tol=1e-6, abs_tol=1e-6)
    if w["space"] == "probabilité":
        assert abs(w["output"] - r["proba"]) < 0.02
    else:
        assert abs(1 / (1 + math.exp(-w["output"])) - r["proba"]) < 0.02


def test_local_custom_record(prepared):
    form = ok(c.get("/api/explain/form", headers=H))
    rec = {f["name"]: f["default"] for f in form["fields"]}
    rec["tenure"] = 2
    rec["Contract"] = "Month-to-month"
    r = ok(c.post("/api/explain/local", json={"model": "xgb", "record": rec}, headers=H))
    assert 0 <= r["proba"] <= 1


def test_sensitive(prepared):
    r = ok(c.get("/api/explain/sensitive/xgb", headers=H))
    names = {f["feature"] for f in r["features"]}
    assert {"gender", "SeniorCitizen"} <= names
    # régression : la part SHAP doit porter sur toutes les colonnes, pas seulement le top 25
    g = ok(c.get("/api/explain/global/xgb", headers=H))["shap"]["share_by_orig"]
    assert math.isclose(sum(g.values()), 1.0, rel_tol=1e-6)
    sc = next(f for f in r["features"] if f["feature"] == "SeniorCitizen")
    assert math.isclose(sc["shap_share"], g["SeniorCitizen"])


def test_errors():
    r = c.post("/api/explain/local", json={"model": "xgb", "test_index": 0}, headers={"X-Session-Id": "vide"})
    assert r.status_code == 400
