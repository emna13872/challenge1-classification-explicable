"""Le garde-fou de l'agent : chaque nombre de la réponse doit venir d'un outil (ou de la question).

Règle volontairement stricte et simple à expliquer :
  - on extrait tous les nombres de la réponse (34,8 % ; 0.348 ; 11 000 ; 1 859…) ;
  - un nombre est « appuyé » s'il est égal, à l'arrondi affiché près, à un nombre renvoyé par un outil
    (« 35 % » est appuyé par 34,8 ; « 34 % » ne l'est pas) ou à un nombre de la question (le n° du client) ;
  - aucune conversion n'est tentée (pas de ×100) : les outils donnent déjà les pourcentages, et une conversion
    autorisée laisserait passer des erreurs de sens (un coût de 0,548 présenté comme « 54,8 % »).
Ce que le garde-fou NE prouve PAS : qu'un nombre appuyé est employé dans le bon sens. Cela, c'est le travail du
banc d'essai (phase E), avec des questions dont on connaît la réponse.
"""
import re
import unicodedata

_LIST = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")              # « 1. », « 2) », « - » en début de ligne
_TOKEN = re.compile(r"(?<![\w.,])(\d{1,3}(?:[   ]\d{3})+(?!\d)|\d+)(?:([.,])(\d+))?(\s?%)?")


def numbers(text: str) -> list[dict]:
    """Nombres d'un texte : [{"text", "values" (interprétations possibles), "decimals"}]."""
    out = []
    for m in _TOKEN.finditer(_LIST.sub(" ", text or "")):
        whole, sep, dec, pct = m.group(1), m.group(2), m.group(3), m.group(4)
        intpart = re.sub(r"[   ]", "", whole)
        values, d = [float(f"{intpart}.{dec}") if dec else float(intpart)], len(dec) if dec else 0
        if sep == "," and dec and len(dec) == 3 and len(intpart) <= 3 and not pct:
            values.append(float(intpart + dec))                    # « 11,000 » à l'anglaise = onze mille
        out.append({"text": m.group(0).strip(), "values": values, "decimals": d})
    return out


def collect(obj, source: str, into: list | None = None, key: str = "") -> list:
    """Tous les nombres d'un résultat d'outil (valeurs numériques ET nombres écrits dans les textes).

    Chaque nombre garde le nom du champ d'où il vient (ex. « seuil_pct ») : c'est ce qui permet de vérifier
    qu'un nombre est attaché au bon mot (voir `misplaced`).
    """
    into = [] if into is None else into
    if isinstance(obj, bool) or obj is None:
        return into
    if isinstance(obj, (int, float)):
        into.append((abs(float(obj)), source, key))
    elif isinstance(obj, str):
        for t in numbers(obj):
            into.extend((v, source, key) for v in t["values"])
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect(v, source, into, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect(v, source, into, key)
    return into


# Règle de voisinage : un pourcentage écrit juste après « seuil » doit être un seuil renvoyé par un outil, un
# pourcentage juste après « risque » doit être un risque renvoyé par un outil (champ dont le nom contient le mot).
ANCHORS = ("seuil", "risque")
_ANCHOR = re.compile(r"\b(seuils?|risques?)\b([^.;!?\n]{0,40}?)(?<![\w.,])(\d+(?:[.,]\d+)?)\s?%")


def misplaced(answer: str, allowed: list) -> list[dict]:
    """Nombres vrais mais attachés au mauvais mot : « le seuil de 34,8 % » alors que 34,8 est le risque."""
    out = []
    for m in _ANCHOR.finditer(_norm(answer or "")):
        word = "seuil" if m.group(1).startswith("seuil") else "risque"
        ok_values = [a[0] for a in allowed if len(a) > 2 and word in a[2]]
        if not ok_values:                                          # aucun outil n'a donné ce genre de valeur
            continue
        dec = m.group(3).replace(",", ".")
        x, d = float(dec), len(dec.split(".")[1]) if "." in dec else 0
        if not any(abs(x - v) <= 0.5 * 10 ** (-d) + 1e-9 for v in ok_values):
            out.append({"word": word, "text": m.group(3) + " %",
                        "expected": sorted({round(v, 1) for v in ok_values})})
    return out


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


_SENSITIVE = re.compile(r"\b(age|sexe|femmes?|hommes?|marie(e|s|es)?|celibataires?|situation (maritale|familiale))\b")
# promesse = futur affirmatif (« vous serez accepté ») ou garantie ; le conditionnel (« seriez ») reste une hypothèse
# honnête, et une garantie NIÉE (« il ne garantit rien », « aucune garantie ») est justement l'avertissement attendu
_PROMISE = re.compile(r"\b(garanti\w*|assure(e|s|es)? d.etre accepte|ser(a|as|ez|ons|ont) (forcement |surement |alors |donc )?accepte\w*)\b")
_NEGATED = re.compile(r"\b(ne|n.)\s*(\w+\s+){0,2}garanti\w*|\b(aucune|pas de|sans)\s+garantie\w*")


def verify(answer: str, allowed: list[tuple[float, str]]) -> dict:
    found = []
    for t in numbers(answer):
        tol = 0.5 * 10 ** (-t["decimals"]) + 1e-9
        src = next((a[1] for x in t["values"] for a in allowed if abs(x - a[0]) <= tol), None)
        found.append({"text": t["text"], "value": t["values"][0], "supported": src is not None, "source": src})
    low = _norm(answer)
    warnings = []
    if _SENSITIVE.search(low):
        warnings.append("La réponse mentionne une variable sensible (âge, sexe, situation maritale…) : elle ne doit "
                        "jamais être présentée comme un motif.")
    if _PROMISE.search(_NEGATED.sub(" ", low)):
        warnings.append("La réponse semble promettre une acceptation : un conseil ne garantit rien.")
    wrong = misplaced(answer, allowed)
    return {"numbers": found, "unsupported": [f["text"] for f in found if not f["supported"]], "misplaced": wrong,
            "n_numbers": len(found), "n_supported": sum(f["supported"] for f in found), "warnings": warnings}
