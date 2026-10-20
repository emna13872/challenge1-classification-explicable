"""Module 1 – chargement automatique et exploration des données."""
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

TELCO_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
             "master/data/Telco-Customer-Churn.csv")

# Dossier des jeux de données locaux (non versionnés : trop volumineux, licences de redistribution).
# Modifiable par la variable d'environnement CREDITLENS_DATA_DIR.
DATA_DIR = Path(os.environ.get("CREDITLENS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

PRESETS = {
    "taiwan_credit": {
        "label": "Défaut de carte de crédit (Taïwan, 2005) – défaut de paiement le mois suivant",
        "target": "default", "positive": "1", "drop": ["ID"],
        "source": "UCI n°350, fichier local (CC BY 4.0)",
        "files": ["default of credit card clients.xls", "default_of_credit_card_clients.csv",
                  "UCI_Credit_Card.csv"],
    },
    "ulb_fraud": {
        "label": "Fraude par carte bancaire (ULB, 2013) – transaction frauduleuse",
        "target": "Class", "positive": "1", "drop": [],
        "source": "ULB / Worldline, fichier local creditcard.csv",
        "files": ["creditcard.csv"],
    },
    "telco": {
        "label": "Telco Customer Churn (IBM) – résiliation d'un client télécom",
        "target": "Churn", "positive": "Yes", "drop": ["customerID"],
        "source": "GitHub IBM (CSV) avec repli sur OpenML",
    },
    "bank_marketing": {
        "label": "Bank Marketing (UCI) – souscription au dépôt à terme",
        "target": "y", "positive": "yes", "drop": ["duration"],
        "source": "UCI via ucimlrepo (variable « duration » retirée)",
    },
    "online_shoppers": {
        "label": "Online Shoppers Purchasing Intention (UCI) – achat en session",
        "target": "Revenue", "positive": "True", "drop": [],
        "source": "UCI via ucimlrepo",
    },
}


def _numeric_coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les chaînes et convertit en numérique les colonnes qui en ont l'air."""
    df = df.copy()
    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
                or pd.api.types.is_datetime64_any_dtype(df[c])):   # object, string, str (pandas 3)
            s = df[c].astype("string").str.strip()
            s = s.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "?": pd.NA})
            conv = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() > 0 and conv.notna().sum() >= 0.9 * s.notna().sum():
                df[c] = conv
            else:
                df[c] = s.astype(object).where(s.notna(), np.nan)
    return df


def _find_local(files: list[str]) -> Path:
    for f in files:
        p = DATA_DIR / f
        if p.exists():
            return p
    # tolérance : le fichier peut être rangé dans un sous-dossier de data/ (un seul niveau)
    for f in files:
        for p in sorted(DATA_DIR.glob(f"*/{f}")):
            return p
    raise FileNotFoundError(
        f"Fichier introuvable dans {DATA_DIR} (attendu : {' ou '.join(files)}). "
        "Voir data/README.md pour le téléchargement.")


# Libellés des codes catégoriels du jeu de Taïwan. Les codes 0, 5 et 6 de EDUCATION et 0 de MARRIAGE
# n'existent pas dans la documentation UCI : on les regroupe dans « others » plutôt que d'inventer un sens.
_TW_SEX = {1: "male", 2: "female"}
_TW_EDU = {1: "graduate school", 2: "university", 3: "high school", 4: "others", 0: "others",
           5: "others", 6: "others"}
_TW_MAR = {1: "married", 2: "single", 3: "others", 0: "others"}


def prepare_taiwan(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage du jeu de Taïwan : noms de colonnes cohérents, codes -> libellés, cible renommée."""
    df = df.copy()
    # Selon la source, la cible s'écrit « default payment next month » ou « default.payment.next.month ».
    for c in list(df.columns):
        if re.sub(r"[^a-z]", "", str(c).lower()) == "defaultpaymentnextmonth":
            df = df.rename(columns={c: "default"})
    # PAY_0 -> PAY_1 : le fichier saute de PAY_0 à PAY_2 ; on rétablit la suite PAY_1..PAY_6 (sept. -> avril).
    if "PAY_0" in df.columns and "PAY_1" not in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})
    for col, mapping in (("SEX", _TW_SEX), ("EDUCATION", _TW_EDU), ("MARRIAGE", _TW_MAR)):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").map(mapping).astype(object)
    df["default"] = pd.to_numeric(df["default"], errors="coerce").astype("Int64").astype(str)
    return df


def _read_local(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path, header=1)        # la 1re ligne du fichier UCI contient X1..X23
    return pd.read_csv(path)


def _from_uci(uci_id: int) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=uci_id)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def load_preset(key: str):
    if key not in PRESETS:
        raise ValueError(f"Jeu de données inconnu : {key}")
    p = PRESETS[key]
    if "files" in p:
        df = _read_local(_find_local(p["files"]))
        if key == "taiwan_credit":
            df = prepare_taiwan(df)
    elif key == "telco":
        try:
            df = pd.read_csv(TELCO_URL)
        except Exception:
            from sklearn.datasets import fetch_openml
            df = fetch_openml(data_id=42178, as_frame=True).frame
    elif key == "bank_marketing":
        df = _from_uci(222)
    else:
        df = _from_uci(468)
    df = _numeric_coerce(df)
    df = df.drop(columns=[c for c in p["drop"] if c in df.columns])
    return df, p["target"], p["positive"]


def load_csv(content: bytes, target: str | None = None):
    text = None
    for enc in ("utf-8", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df = _numeric_coerce(df)
    if target is None or target not in df.columns:
        target = df.columns[-1]
    return df, target


def binarize(series: pd.Series, positive) -> pd.Series:
    return (series.astype(str) == str(positive)).astype(int)


def default_positive(series: pd.Series):
    """Classe minoritaire par défaut (celle d'intérêt)."""
    vc = series.dropna().astype(str).value_counts()
    return vc.index[-1] if len(vc) > 1 else vc.index[0]


def column_info(df: pd.DataFrame, target: str):
    cols = []
    for c in df.columns:
        s = df[c]
        n = len(s)
        nun = int(s.nunique(dropna=True))
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        id_like = (not numeric and nun > 0.9 * n and n > 50) or bool(
            re.search(r"(^id$|_id$|^id_|customerid|uuid)", c.lower()))
        cols.append({
            "name": c, "kind": "numeric" if numeric else "categorical",
            "n_unique": nun, "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "suggest_exclude": bool(id_like and c != target),
        })
    return cols


def summary(df: pd.DataFrame, target: str, positive, name: str):
    num = df.select_dtypes("number")
    desc = []
    if not num.empty:
        d = num.describe().T
        d["missing"] = num.isna().sum()
        d = d.round(3).reset_index().rename(columns={"index": "column"})
        desc = d.to_dict(orient="records")
    vc = df[target].astype(str).value_counts()
    classes = [{"label": k, "count": int(v), "pct": float(v / len(df) * 100)} for k, v in vc.items()]
    pos_n = int(vc.get(str(positive), 0))
    return {
        "name": name, "n_rows": len(df), "n_cols": df.shape[1],
        "target": target, "positive": str(positive),
        "classes": classes, "positive_rate": pos_n / len(df) if len(df) else 0,
        "imbalanced": bool(len(df) and (pos_n / len(df) < 0.3 or pos_n / len(df) > 0.7)),
        "multiclass": bool(len(vc) > 2),
        "columns": column_info(df, target),
        "describe": desc,
        "preview": df.head(20).astype(object).where(df.head(20).notna(), None).to_dict(orient="records"),
    }


def column_view(df, col, target, positive, bins=20):
    """Distribution de la colonne et taux de la classe positive par modalité / tranche."""
    y = binarize(df[target], positive)
    s = df[col]
    out = {"column": col}
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        valid = s.notna()
        counts, edges = np.histogram(s[valid], bins=min(bins, 30))
        out["kind"] = "numeric"
        out["hist"] = [{"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
                       for i in range(len(counts))]
        q = pd.qcut(s[valid], q=min(10, s[valid].nunique()), duplicates="drop")
        g = y[valid].groupby(q, observed=True).agg(["mean", "count"])
        out["rate"] = [{"label": f"{iv.left:.4g} – {iv.right:.4g}", "rate": float(r["mean"]),
                        "count": int(r["count"])} for iv, r in g.iterrows()]
    else:
        out["kind"] = "categorical"
        ss = s.astype(object).where(s.notna(), "(manquant)").astype(str)
        g = y.groupby(ss).agg(["mean", "count"]).sort_values("count", ascending=False).head(30)
        out["hist"] = [{"label": k, "count": int(r["count"])} for k, r in g.iterrows()]
        out["rate"] = [{"label": k, "rate": float(r["mean"]), "count": int(r["count"])}
                       for k, r in g.iterrows()]
    out["base_rate"] = float(y.mean())
    return out


def correlation(df, target, positive, max_cols=25):
    num = df.select_dtypes("number").copy()
    num["__cible__"] = binarize(df[target], positive)
    if num.shape[1] > max_cols:
        c = num.corr()["__cible__"].abs().sort_values(ascending=False)
        num = num[list(c.index[:max_cols])]
    corr = num.corr().fillna(0)
    return {"labels": [("cible" if c == "__cible__" else c) for c in corr.columns],
            "matrix": corr.round(3).values.tolist(),
            "target_corr": [{"feature": c, "corr": float(corr.loc[c, "__cible__"])}
                            for c in corr.columns if c != "__cible__"]}
