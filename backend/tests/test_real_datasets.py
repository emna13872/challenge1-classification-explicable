"""Jeux réels de crédit et de fraude.

Deux familles de tests :
- tests unitaires sur un mini-tableau fabriqué à la main : ils tournent partout (CI comprise) ;
- tests d'intégration sur le vrai fichier : ignorés (« skip ») si le fichier n'est pas présent.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import data as D
from app.explain import is_sensitive
from app.main import app

H = {"X-Session-Id": "pytest-real"}
c = TestClient(app)


def _mini_taiwan() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "SEX": [1, 2, 2, 1, 2],
        "EDUCATION": [1, 2, 0, 5, 6],        # 0, 5, 6 : codes non documentés
        "MARRIAGE": [1, 2, 3, 0, 2],         # 0 : code non documenté
        "PAY_0": [2, -1, 0, -2, 1],
        "default payment next month": [1, 0, 0, 1, 0],
    })


def test_prepare_taiwan_labels_and_names():
    out = D.prepare_taiwan(_mini_taiwan())
    assert "PAY_1" in out.columns and "PAY_0" not in out.columns
    assert "default" in out.columns and "default payment next month" not in out.columns
    assert out["SEX"].tolist() == ["male", "female", "female", "male", "female"]
    # les codes inconnus sont regroupés, jamais inventés
    assert out["EDUCATION"].tolist() == ["graduate school", "university", "others", "others", "others"]
    assert out["MARRIAGE"].tolist() == ["married", "single", "others", "others", "single"]
    assert out["default"].tolist() == ["1", "0", "0", "1", "0"]


def test_prepare_taiwan_accepts_kaggle_column_spelling():
    df = _mini_taiwan().rename(columns={"default payment next month": "default.payment.next.month"})
    assert "default" in D.prepare_taiwan(df).columns


@pytest.mark.parametrize("col", ["SEX", "AGE", "MARRIAGE"])
def test_sensitive_columns_detected(col):
    assert is_sensitive(col)


def test_missing_file_gives_clear_404(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "DATA_DIR", tmp_path)
    r = c.post("/api/data/load", json={"dataset": "ulb_fraud"}, headers=H)
    assert r.status_code == 404 and "creditcard.csv" in r.json()["detail"]


# ------------------------------------------------------------- intégration (fichier réel requis)
taiwan_present = any((D.DATA_DIR / f).exists() for f in D.PRESETS["taiwan_credit"]["files"])
fraud_present = (D.DATA_DIR / "creditcard.csv").exists()


@pytest.mark.skipif(not taiwan_present, reason="fichier Taïwan absent de data/")
def test_taiwan_real_file():
    r = c.post("/api/data/load", json={"dataset": "taiwan_credit"}, headers=H)
    assert r.status_code == 200, r.text
    s = r.json()
    assert (s["n_rows"], s["target"]) == (30000, "default")
    assert abs(s["positive_rate"] - 0.2212) < 0.0005          # 6 636 défauts sur 30 000
    kinds = {col["name"]: col["kind"] for col in s["columns"]}
    assert kinds["SEX"] == kinds["EDUCATION"] == kinds["MARRIAGE"] == "categorical"
    assert kinds["PAY_1"] == kinds["LIMIT_BAL"] == "numeric"
    assert "ID" not in kinds
    assert sum(col["missing"] for col in s["columns"]) == 0


@pytest.mark.skipif(not fraud_present, reason="creditcard.csv absent de data/")
def test_fraud_real_file():
    r = c.post("/api/data/load", json={"dataset": "ulb_fraud"}, headers=H)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["n_rows"] == 284807 and s["target"] == "Class"
    assert round(s["positive_rate"] * s["n_rows"]) == 492     # 492 fraudes (0,172 %)
    assert s["imbalanced"]


@pytest.mark.skipif(not taiwan_present, reason="fichier Taïwan absent de data/")
def test_taiwan_end_to_end():
    """Le jeu réel traverse tout le pipeline : préparation, modèles, évaluation, SHAP/LIME, équité."""
    s = c.post("/api/data/load", json={"dataset": "taiwan_credit"}, headers=H).json()
    feats = [col["name"] for col in s["columns"] if col["name"] != "default" and not col["suggest_exclude"]]
    p = c.post("/api/prep", json={"features": feats}, headers=H)
    assert p.status_code == 200, p.text
    t = c.post("/api/models/train", json={"models": [{"name": "logreg"}, {"name": "xgb"}]}, headers=H)
    assert t.status_code == 200, t.text
    ev = c.get("/api/evaluate?threshold=0.3", headers=H).json()
    assert 0.70 < ev["models"]["xgb"]["auc"] < 0.85           # littérature : environ 0,77 à 0,79
    loc = c.post("/api/explain/local", json={"model": "xgb", "test_index": 0}, headers=H)
    assert loc.status_code == 200, loc.text
    sens = c.get("/api/explain/sensitive/xgb", headers=H)
    assert sens.status_code == 200, sens.text
    assert {"SEX", "AGE", "MARRIAGE"} <= {f["feature"] for f in sens.json()["features"]}
