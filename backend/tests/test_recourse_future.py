"""Recours en mode « avenir » : le passé n'est jamais réécrit, chaque conseil est un comportement à venir, vérifié.

Données synthétiques au format de Taïwan (mêmes noms de colonnes), fabriquées ici : tournent partout, CI comprise.
"""
import numpy as np
import pandas as pd
import pytest

from app import explain as X
from app import models as MD
from app import prep as P
from app import recourse as RC
from app import recourse_audit as RA
from app.state import Session

THR = 0.25
W = RC.TIME_WINDOWS["taiwan_credit"]


def _monthly(n=2000, seed=0) -> pd.DataFrame:
    """Le risque monte avec les retards récents et baisse quand on rembourse une bonne part de sa facture."""
    rng = np.random.default_rng(seed)
    d = {"LIMIT_BAL": rng.choice([20000, 50000, 100000, 200000], n), "AGE": rng.integers(21, 70, n),
         "SEX": rng.choice(["female", "male"], n)}
    late = rng.random(n) < 0.35
    logit = -1.6 + 0.0 * d["AGE"]
    for k in range(1, 7):
        delay = np.where(late & (rng.random(n) < 0.6), rng.integers(1, 3, n), rng.choice([-1, 0], n))
        bill = np.where(rng.random(n) < 0.08, 0, rng.gamma(2.0, 15000, n)).round()
        paid = np.where(bill > 0, np.minimum(bill, rng.gamma(1.5, 2000, n)), 0).round()
        d[f"PAY_{k}"], d[f"BILL_AMT{k}"], d[f"PAY_AMT{k}"] = delay, bill, paid
        w = 1.0 / k                                                    # les mois récents pèsent plus
        logit = logit + w * (0.9 * (delay > 0) - 0.00025 * paid)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({**d, "default": y.astype(str)})


@pytest.fixture(scope="module")
def s():
    s = Session()
    s.df, s.target, s.positive, s.name = _monthly(), "default", "1", "taiwan_credit"
    P.build(s, {"features": [c for c in s.df.columns if c != "default"], "test_size": 0.25, "random_state": 0,
                "num_impute": "median", "cat_impute": "mode", "scaling": "standard", "stratify": True})
    est, info = MD.train_one("xgb", {}, False, False, s.Z_train, s.y_train.values)
    s.models["xgb"] = {"est": est, "info": info, "p_test": est.predict_proba(s.Z_test)[:, 1], "label": "xgb"}
    return s


def _refused(s):
    return [int(i) for i in np.where(s.models["xgb"]["p_test"] >= THR)[0]]


def test_the_window_shifts_and_the_past_is_never_rewritten():
    raw = {"LIMIT_BAL": 50000, "AGE": 30, "SEX": "female"}
    for k in range(1, 7):
        raw[f"PAY_{k}"], raw[f"BILL_AMT{k}"], raw[f"PAY_AMT{k}"] = 10 + k, 1000 * k, 100 * k
    new = RC._shifted(raw, W, 2, payment=500, bill=1000)
    for k in (3, 4, 5, 6):                                             # l'ancien mois k-2 devient le mois k
        assert new[f"PAY_{k}"] == raw[f"PAY_{k - 2}"] and new[f"PAY_AMT{k}"] == raw[f"PAY_AMT{k - 2}"]
        assert new[f"BILL_AMT{k}"] == raw[f"BILL_AMT{k - 2}"]
    for k in (1, 2):                                                   # les 2 mois nouveaux : sans retard
        assert (new[f"PAY_{k}"], new[f"BILL_AMT{k}"], new[f"PAY_AMT{k}"]) == (W["code_partial"], 1000, 500)
    assert {c: new[c] for c in ("LIMIT_BAL", "AGE", "SEX")} == {"LIMIT_BAL": 50000, "AGE": 30, "SEX": "female"}
    assert raw["PAY_1"] == 11                                          # l'original n'est pas modifié
    full = RC._shifted(raw, W, 6, payment=1000, bill=1000)
    assert all(full[f"PAY_{k}"] == W["code_full"] for k in range(1, 7))   # 6 mois : fenêtre entièrement neuve
    none = RC._shifted(raw, W, 1, payment=0, bill=0)
    assert none["PAY_1"] == W["code_none"] and none["PAY_2"] == raw["PAY_1"]


def test_payment_candidates_are_realistic():
    ref = np.r_[np.zeros(200), np.arange(1, 1001) * 10.0]
    c = RC.payment_candidates(ref, bill=4000)
    assert c == sorted(c) and c[0] > 0 and c[-1] <= 4000                # jamais 0 (ce serait un retard), jamais > facture
    assert 4000 in c                                                   # payer la facture en entier est essayé
    big = RC.payment_candidates(ref, bill=10 ** 7)
    assert max(big) <= RC._nice(float(np.quantile(ref, 0.95)), True, 2)   # au plus le 95e centile, même si la facture est énorme
    assert RC.payment_candidates(ref, bill=0) == [0.0]                 # rien à payer : un seul scénario


def test_taiwan_uses_the_future_mode_by_default(s):
    i = _refused(s)[0]
    assert RC.recourse(s, "xgb", THR, test_index=i)["mode"] == "future"
    assert RC.recourse(s, "xgb", THR, test_index=i, actionable={"PAY_AMT1": "up"})["mode"] == "now"
    assert "taiwan_credit" not in RC.ACTIONABLE                         # aucune variable de Taïwan ne change « maintenant »


def test_every_option_is_verified_independently_through_the_full_pipeline(s):
    est = s.models["xgb"]["est"]
    seen = 0
    for i in _refused(s)[:40]:
        r = RC.recourse(s, "xgb", THR, test_index=i)
        raw = X.client_matrices(s, test_index=i)[2]
        for o in r["options"]:
            seen += 1
            rec = RC._shifted(raw, W, o["horizon_months"], o["monthly_payment"], r["assumed_bill"])
            _, Z, _, _ = X.client_matrices(s, record=rec)
            assert abs(float(est.predict_proba(Z)[0, 1]) - o["new_probability"]) < 1e-6
            assert o["new_probability"] < THR <= r["probability"]
    assert seen > 0, "le jeu synthétique doit produire des conseils"


def test_each_advice_is_the_smallest_amount_that_works(s):
    est = s.models["xgb"]["est"]
    for i in _refused(s)[:25]:
        r = RC.recourse(s, "xgb", THR, test_index=i)
        if r["status"] != "found":
            continue
        raw = X.client_matrices(s, test_index=i)[2]
        cands = RC.payment_candidates(s.X_train_raw["PAY_AMT1"].to_numpy(float), r["assumed_bill"])
        hs = [o["horizon_months"] for o in r["options"]]
        assert hs == sorted(hs) and len(set(hs)) == len(hs)            # au plus un conseil par délai, du plus court au plus long
        for o in r["options"]:
            assert 0 < o["monthly_payment"] <= r["assumed_bill"] or r["assumed_bill"] == 0
            smaller = [c for c in cands if c < o["monthly_payment"]]
            for c in smaller:                                          # aucun montant plus petit ne suffisait
                rec = RC._shifted(raw, W, o["horizon_months"], c, r["assumed_bill"])
                assert float(est.predict_proba(X.client_matrices(s, record=rec)[1])[0, 1]) >= THR


def test_text_states_numbers_and_assumptions(s):
    r = next(r for r in (RC.recourse(s, "xgb", THR, test_index=i) for i in _refused(s)) if r["status"] == "found")
    o = r["options"][0]
    assert RC._pct(r["probability"], THR) in o["text"] and RC._pct(o["new_probability"], THR) in o["text"]
    assert f"{o['horizon_months']} mois" in o["text"] and "Nouvelle demande" in o["text"]
    assert len(r["assumptions"]) == 3 and "factures" in r["assumptions"][0]
    assert "ne garantit" in r["caveat"]


def test_honest_statuses(s):
    low = int(np.argmin(s.models["xgb"]["p_test"]))
    assert RC.recourse(s, "xgb", 0.5, test_index=low)["status"] == "already_accepted"
    i = _refused(s)[0]
    low = RC.recourse(s, "xgb", THR, test_index=i)["lowest_probability_found"]
    hard = RC.recourse(s, "xgb", low * 0.999, test_index=i)            # seuil juste sous le mieux atteignable
    assert hard["status"] == "none_found" and hard["options"] == []
    assert abs(hard["lowest_probability_found"] - low) < 1e-12           # on dit jusqu'où on est descendu
    feats, s.features = s.features, [f for f in s.features if f != "BILL_AMT3"]
    try:                                                               # fenêtre incomplète : on ne devine pas
        assert RC.future_recourse(s, "xgb", THR, test_index=0)["status"] == "no_actionable"
    finally:
        s.features = feats


def test_audit_in_future_mode(s):
    a = RA.audit(s, "xgb", THR, max_clients=None)
    assert a["mode"] == "future" and a["status"] == "ok"
    assert a["n_found"] + a["n_none"] == a["n_audited"] == len(_refused(s))
    h = a["effort"]["horizon_months"]
    assert sum(h.values()) == a["n_found"] and set(h) <= {"1", "3", "6"}
    if a["n_none"]:
        assert 0 <= a["none_found"]["share_no_bill"] <= 1
