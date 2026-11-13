"""Fonctions de nettoyage du Global Shark Attack File (jeu Kaggle « Shark Attacks »).

Ce module est importé par le notebook etape1_requins.ipynb ET doit être réutilisé
tel quel lors du déploiement : une saisie brute passe par `preparer_variables`,
puis par le pipeline scikit-learn sauvegardé (qui contient `RegroupeurModalitesRares`).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
INCONNU = "inconnu"
AUTRE = "AUTRE"

# Types d'incident conservés (attaques confirmées impliquant baigneurs / usagers de la mer)
TYPES_CONSERVES = {"Unprovoked": "non_provoquee", "Provoked": "provoquee",
                   "Watercraft": "embarcation", "Boat": "embarcation"}

ANNEE_MIN_VALIDE, ANNEE_MAX_VALIDE = 1500, 2025
AGE_MAX_VALIDE = 100

# Variables d'entrée du modèle (après préparation)
VARIABLES_NUMERIQUES = ["annee", "age"]
VARIABLES_CATEGORIELLES = ["type_attaque", "pays", "region", "activite_cat",
                           "espece_cat", "sexe_cat", "moment_journee"]
VARIABLES_MODELE = VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES


def _est_vide(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""


# --------------------------------------------------------------------------
# Cible
# --------------------------------------------------------------------------
def nettoyer_cible(x):
    """'Y'/'y'/'Y x 2' -> 1 ; 'N'/'n' -> 0 ; tout le reste (UNKNOWN, F, Nq, 2017.0...) -> NaN.

    'F', 'Nq' et '2017.0' sont ambigus : on préfère les exclure plutôt que deviner.
    """
    if _est_vide(x):
        return np.nan
    s = str(x).strip().upper()
    if s in {"Y", "Y X 2"}:
        return 1
    if s == "N":
        return 0
    return np.nan


# --------------------------------------------------------------------------
# Type d'attaque
# --------------------------------------------------------------------------
def nettoyer_type(x) -> str | None:
    """Retourne la catégorie française du type d'incident, ou None s'il est hors périmètre."""
    if _est_vide(x):
        return None
    return TYPES_CONSERVES.get(str(x).strip())


# --------------------------------------------------------------------------
# Année
# --------------------------------------------------------------------------
def nettoyer_annee(x):
    """Conversion numérique ; les valeurs hors [1500, 2025] (ex. 1, 1000) deviennent NaN."""
    a = pd.to_numeric(x, errors="coerce")
    if pd.isna(a) or a < ANNEE_MIN_VALIDE or a > ANNEE_MAX_VALIDE:
        return np.nan
    return float(int(a))


# --------------------------------------------------------------------------
# Âge
# --------------------------------------------------------------------------
_RE_NOMBRE = re.compile(r"\d+(?:\.\d+)?")


def nettoyer_age(x):
    """Extrait un âge (en années) d'un texte libre. Règles, dans l'ordre :

    1. vide -> NaN ;
    2. plusieurs victimes ('&', ',', 'and', 'both') ou caractères douteux ('!', '<', '>') -> NaN ;
    3. 'teen' / 'teens' -> 15 ;
    4. durée en mois ('18 months', '2 to 3 months') -> moyenne / 12 ;
    5. décennie ('20s', "20's", 'mid-30s') -> milieu de la décennie (25, 35) ;
    6. intervalle ('30 or 40', '16 to 18') -> moyenne des deux bornes ;
    7. un seul nombre ('17', '17.0', 'Ca. 33', '20?', '2½') -> ce nombre ;
    8. termes vagues ('adult', 'young', 'elderly', 'a minor'...) -> NaN ;
    9. résultat hors [0, 100] -> NaN.
    """
    if _est_vide(x):
        return np.nan
    s = str(x).strip().lower().replace("½", ".5")
    if re.search(r"[&,!<>]|\band\b|\bboth\b", s):
        return np.nan
    if re.search(r"\bteens?\b", s):
        return 15.0
    nombres = [float(n) for n in _RE_NOMBRE.findall(s)]
    if not nombres:
        return np.nan
    if "month" in s:
        age = np.mean(nombres) / 12
    elif re.search(r"\d+\s*'?s\b", s):
        age = nombres[0] + 5
    elif len(nombres) == 2 and re.search(r"\bor\b|\bto\b|-", s):
        age = np.mean(nombres)
    elif len(nombres) == 1:
        age = nombres[0]
    else:
        return np.nan
    return float(age) if 0 <= age <= AGE_MAX_VALIDE else np.nan


def tranche_age(age) -> str:
    """Tranche d'âge (pour l'EDA)."""
    if pd.isna(age):
        return INCONNU
    if age < 18:
        return "0-17"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


# --------------------------------------------------------------------------
# Sexe
# --------------------------------------------------------------------------
def nettoyer_sexe(x) -> str:
    """'M', ' M', 'M x 2' -> 'M' ; 'F' -> 'F' ; le reste ('N', '.', 'lli', vide) -> 'inconnu'."""
    if _est_vide(x):
        return INCONNU
    s = str(x).strip().upper()
    if re.fullmatch(r"M( X \d+)?", s):
        return "M"
    if re.fullmatch(r"F( X \d+)?", s):
        return "F"
    return INCONNU


# --------------------------------------------------------------------------
# Heure -> moment de la journée
# --------------------------------------------------------------------------
_RE_HEURE = re.compile(r"(?<!\d)(\d{1,2})[hj:]\d{0,2}|(?<!\d)(\d{2})(\d{2})(?!\d)")


def _moment_depuis_heure(h: int) -> str:
    if 5 <= h <= 11:
        return "matin"
    if 12 <= h <= 16:
        return "apres_midi"
    if 17 <= h <= 20:
        return "soir"
    return "nuit"  # 21h-4h


def moment_journee(x) -> str:
    """Moment de la journée : matin (5-11h), après-midi (12-16h), soir (17-20h), nuit (21-4h).

    On lit d'abord une heure explicite ('14h30', '1600', '10j30', '11hoo') ; à défaut,
    des mots-clés ('morning', 'dusk', 'night'...). Sinon 'inconnu'.
    """
    if _est_vide(x):
        return INCONNU
    s = str(x).strip().lower()
    m = _RE_HEURE.search(s)
    if m:
        h = int(m.group(1) or m.group(2))
        if 0 <= h <= 23:
            return _moment_depuis_heure(h)
    if re.search(r"afternoon|afternon|midday|noon|lunch|\bp\.?m\.?\b", s):
        return "apres_midi"
    if re.search(r"morning|dawn|\ba\.?m\.?\b", s):
        return "matin"
    if re.search(r"evening|dusk|sunset|sundown", s):
        return "soir"
    if re.search(r"night|dark|midnight", s):
        return "nuit"
    return INCONNU


# --------------------------------------------------------------------------
# Espèce
# --------------------------------------------------------------------------
_MOTIFS_ESPECE = {
    # 'sand tiger' et 'grey nurse' sont des espèces différentes du requin tigre
    "autre_identifiee": r"sand\s?tiger|grey nurse|ragged\s?tooth|mako|black\s?tip|hammerhead|"
                        r"nurse shark|lemon shark|blue shark|whaler|wobbegong|spinner|reef shark|"
                        r"whitetip|white-tip|sevengill|seven-gill|cow shark|sandbar|dusky|galapagos|"
                        r"silky|porbeagle|angel shark|cookiecutter|carpet shark|dogfish|thresher|"
                        r"basking|leopard shark|horn shark|salmon shark|blacktail|caribbean reef|"
                        r"grey shark|gray shark|\btope\b|school shark|broadnose|shovelnose",
    "requin_blanc": r"white shark|great white|carcharodon|carcharias(?! taurus)",
    "requin_tigre": r"tiger shark|tiger|galeocerdo",
    "requin_bouledogue": r"bull shark|bull|zambe[sz]i|leucas",
}


def espece_principale(x) -> str:
    """Espèce principale = premier motif d'espèce trouvé dans le texte.

    Catégories : requin_blanc, requin_tigre, requin_bouledogue, autre_identifiee, inconnue.
    Un texte sans nom d'espèce ('6' shark', 'Invalid', 'Shark involvement not confirmed')
    donne 'inconnue'.
    """
    if _est_vide(x):
        return "inconnue"
    s = str(x).lower()
    meilleure, pos = "inconnue", len(s) + 1
    for cat, motif in _MOTIFS_ESPECE.items():
        m = re.search(motif, s)
        # en cas d'égalité de position, l'ordre du dictionnaire prime (autre_identifiee d'abord,
        # pour que 'sand tiger' ne soit pas classé en requin tigre)
        if m and m.start() < pos:
            meilleure, pos = cat, m.start()
    return meilleure


# --------------------------------------------------------------------------
# Activité
# --------------------------------------------------------------------------
_MOTIFS_ACTIVITE = [  # ordre = priorité
    ("chasse_sous_marine", r"spear"),
    ("kayak_bateau", r"surf[\s-]?ski|overboard"),
    ("peche", r"fish|angl|net|crabb|lobster"),
    ("plongee", r"div|snorkel|scuba"),
    ("surf", r"surf|board|boogie|kite|\bsup\b"),
    ("kayak_bateau", r"kayak|canoe|boat|\brow|sail|paddl|overboard|ship|yacht|raft|watercraft|ferry|vessel|jet ?ski"),
    ("baignade", r"swim|bath|wad|float|stand|splash|play|tread|jump|walk|dangl|sitting|fell into"),
]


def categorie_activite(x) -> str:
    """Regroupe le texte libre de l'activité en catégories, par mots-clés (premier motif qui
    correspond dans l'ordre de priorité). Vide -> 'inconnue' ; aucun motif -> 'autre'."""
    if _est_vide(x):
        return "inconnue"
    s = str(x).lower()
    for cat, motif in _MOTIFS_ACTIVITE:
        if re.search(motif, s):
            return cat
    return "autre"


# --------------------------------------------------------------------------
# Pays / région
# --------------------------------------------------------------------------
def normaliser_texte_lieu(x) -> str:
    """Majuscules, espaces multiples réduits, '?' et espaces en bord supprimés. Vide -> 'INCONNU'."""
    if _est_vide(x):
        return INCONNU.upper()
    s = re.sub(r"\s+", " ", str(x)).strip().strip("?").strip().upper()
    return s or INCONNU.upper()


class RegroupeurModalitesRares(BaseEstimator, TransformerMixin):
    """Remplace par 'AUTRE' les modalités vues moins de `min_effectif` fois À L'APPRENTISSAGE.

    Placé dans le pipeline : la liste des modalités fréquentes est apprise sur chaque pli
    d'entraînement (pas de fuite), et une modalité jamais vue (ex. nouveau pays en
    production) devient 'AUTRE'.
    """

    def __init__(self, min_effectif: int = 20):
        self.min_effectif = min_effectif

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.modalites_frequentes_ = {
            c: set(vc[vc >= self.min_effectif].index) for c, vc in
            ((c, X[c].value_counts()) for c in X.columns)
        }
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for c in X.columns:
            X[c] = X[c].where(X[c].isin(self.modalites_frequentes_[c]), AUTRE)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# --------------------------------------------------------------------------
# Préparation complète
# --------------------------------------------------------------------------
def preparer_variables(df_brut: pd.DataFrame) -> pd.DataFrame:
    """Applique tous les nettoyages à un DataFrame au format brut du CSV
    (colonnes year, type, country, area, activity, sex, age, time, species).
    Retourne uniquement les variables d'entrée du modèle (VARIABLES_MODELE)."""
    out = pd.DataFrame(index=df_brut.index)
    out["annee"] = df_brut["year"].map(nettoyer_annee)
    out["age"] = df_brut["age"].map(nettoyer_age)
    out["type_attaque"] = df_brut["type"].map(nettoyer_type).fillna(INCONNU)
    out["pays"] = df_brut["country"].map(normaliser_texte_lieu)
    out["region"] = df_brut["area"].map(normaliser_texte_lieu)
    out["activite_cat"] = df_brut["activity"].map(categorie_activite)
    out["espece_cat"] = df_brut["species"].map(espece_principale)
    out["sexe_cat"] = df_brut["sex"].map(nettoyer_sexe)
    out["moment_journee"] = df_brut["time"].map(moment_journee)
    return out[VARIABLES_MODELE]
