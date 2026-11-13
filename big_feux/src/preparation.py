"""Préparation des données FPA FOD (feux de forêt, USA) en expressions Spark SQL natives.

Les transformations sont écrites en SQL Spark (pas d'UDF Python) pour deux raisons :
- performance : elles sont exécutées dans la JVM, sans sérialisation vers Python ;
- déploiement : elles sont encapsulées dans des `SQLTransformer`, étapes natives d'un
  `PipelineModel` sauvegardable. Le modèle déployé applique donc EXACTEMENT ce nettoyage
  à une saisie brute (colonnes du fichier source).
"""
from __future__ import annotations

from pyspark.ml.feature import SQLTransformer
from pyspark.sql import DataFrame, functions as F

# ---------------------------------------------------------------------------
# Colonnes
# ---------------------------------------------------------------------------
# Colonnes lues dans la table SQLite `Fires` (la géométrie `Shape`, binaire, est ignorée :
# elle duplique LATITUDE/LONGITUDE).
COLONNES_SOURCE = [
    "OBJECTID", "FOD_ID", "FPA_ID", "SOURCE_SYSTEM_TYPE", "SOURCE_SYSTEM",
    "NWCG_REPORTING_AGENCY", "NWCG_REPORTING_UNIT_ID", "NWCG_REPORTING_UNIT_NAME",
    "SOURCE_REPORTING_UNIT", "SOURCE_REPORTING_UNIT_NAME", "LOCAL_FIRE_REPORT_ID",
    "LOCAL_INCIDENT_ID", "FIRE_CODE", "FIRE_NAME", "ICS_209_INCIDENT_NUMBER", "ICS_209_NAME",
    "MTBS_ID", "MTBS_FIRE_NAME", "COMPLEX_NAME", "FIRE_YEAR", "DISCOVERY_DATE", "DISCOVERY_DOY",
    "DISCOVERY_TIME", "STAT_CAUSE_CODE", "STAT_CAUSE_DESCR", "CONT_DATE", "CONT_DOY", "CONT_TIME",
    "FIRE_SIZE", "FIRE_SIZE_CLASS", "LATITUDE", "LONGITUDE", "OWNER_CODE", "OWNER_DESCR",
    "STATE", "COUNTY", "FIPS_CODE", "FIPS_NAME",
]

# Entrées brutes nécessaires au modèle (toutes connues au moment de la découverte)
ENTREES_BRUTES = ["DISCOVERY_DATE", "DISCOVERY_TIME", "LATITUDE", "LONGITUDE", "STATE",
                  "OWNER_DESCR", "NWCG_REPORTING_AGENCY", "SOURCE_SYSTEM_TYPE", "NWCG_REPORTING_UNIT_ID"]

INCONNU = "INCONNU"
AUTRE = "AUTRE"
PAS_GRILLE = 1.0                     # degrés (voir justification dans le notebook)
LAT_MIN, LAT_MAX = 17.0, 72.0        # Porto Rico (≈ 17,9° N) à l'Alaska (≈ 71,4° N)
LON_MIN, LON_MAX = -180.0, -64.0     # Alaska / Hawaï à l'est du Maine et Porto Rico
JOUR_JULIEN_EPOCH = 2440587.5        # jour julien du 1970-01-01 à 0 h

# ---------------------------------------------------------------------------
# Cible
# ---------------------------------------------------------------------------
CAUSES_FR = {
    "Debris Burning": "Brûlage de débris", "Miscellaneous": "Divers", "Arson": "Incendie criminel",
    "Lightning": "Foudre", "Equipment Use": "Équipement", "Campfire": "Feu de camp",
    "Children": "Enfants", "Smoking": "Cigarette", "Railroad": "Chemin de fer",
    "Powerline": "Ligne électrique", "Fireworks": "Feux d'artifice",
    "Structure": "Divers",           # 0,2 % des feux : regroupé avec « Divers » (voir notebook)
}
CAUSE_INCONNUE = "Missing/Undefined"


def litteral_sql(texte: str) -> str:
    """Littéral de chaîne Spark SQL. Spark échappe par antislash : 'a''b' serait lu comme 'a' 'b' -> « ab »."""
    return "'" + texte.replace("\\", "\\\\").replace("'", "\\'") + "'"


def expression_cause() -> str:
    """Cause en français ; NULL pour « Missing/Undefined » (cause inconnue, hors apprentissage)."""
    cas = " ".join(f"WHEN {litteral_sql(en)} THEN {litteral_sql(fr)}" for en, fr in CAUSES_FR.items())
    return f"CASE STAT_CAUSE_DESCR {cas} ELSE NULL END"


def ajouter_cause(df: DataFrame) -> DataFrame:
    return df.withColumn("cause", F.expr(expression_cause()))


# ---------------------------------------------------------------------------
# Variables explicatives (expressions SQL)
# ---------------------------------------------------------------------------
_DATE = f"date_add(DATE'1970-01-01', CAST(DISCOVERY_DATE - {JOUR_JULIEN_EPOCH} AS INT))"
_HEURE = ("CASE WHEN DISCOVERY_TIME RLIKE '^[0-9]{4}$' "
          "AND CAST(substr(DISCOVERY_TIME, 1, 2) AS INT) BETWEEN 0 AND 23 "
          "AND CAST(substr(DISCOVERY_TIME, 3, 2) AS INT) BETWEEN 0 AND 59 "
          "THEN CAST(substr(DISCOVERY_TIME, 1, 2) AS INT) END")


def _norm(col: str) -> str:
    """Majuscules, espaces réduits ; NULL ou vide -> 'INCONNU'."""
    return (f"COALESCE(NULLIF(upper(trim(regexp_replace(CAST({col} AS STRING), '\\\\s+', ' '))), ''), "
            f"'{INCONNU}')")


# Étape 1 : variables de base, calculées directement depuis les colonnes brutes
EXPRESSIONS_BASE = {
    "date_decouverte": _DATE,
    "heure": _HEURE,
    "lat": f"CASE WHEN LATITUDE BETWEEN {LAT_MIN} AND {LAT_MAX} THEN CAST(LATITUDE AS DOUBLE) END",
    "lon": f"CASE WHEN LONGITUDE BETWEEN {LON_MIN} AND {LON_MAX} THEN CAST(LONGITUDE AS DOUBLE) END",
    "etat": _norm("STATE"),
    "proprietaire": _norm("OWNER_DESCR"),
    "organisme": _norm("NWCG_REPORTING_AGENCY"),
    "type_source": _norm("SOURCE_SYSTEM_TYPE"),
    "unite": _norm("NWCG_REPORTING_UNIT_ID"),
}

# Étape 2 : variables dérivées des variables de base
EXPRESSIONS_DERIVEES = {
    "annee": "year(date_decouverte)",
    "mois": "month(date_decouverte)",
    "jour_annee": "dayofyear(date_decouverte)",
    "jour_semaine": "dayofweek(date_decouverte)",          # 1 = dimanche ... 7 = samedi
    "week_end": "CASE WHEN dayofweek(date_decouverte) IN (1, 7) THEN 1.0 ELSE 0.0 END",
    "heure_cat": f"COALESCE(lpad(CAST(heure AS STRING), 2, '0'), 'inconnue')",
    "cellule": (f"CASE WHEN lat IS NULL OR lon IS NULL THEN '{INCONNU}' ELSE "
                f"concat(CAST(floor(lat / {PAS_GRILLE}) * {PAS_GRILLE} AS STRING), '_', "
                f"CAST(floor(lon / {PAS_GRILLE}) * {PAS_GRILLE} AS STRING)) END"),
}


def requete_preparation() -> str:
    """Requête SQL (pour SQLTransformer) qui ajoute toutes les variables à __THIS__."""
    base = ", ".join(f"{e} AS {n}" for n, e in EXPRESSIONS_BASE.items())
    deriv = ", ".join(f"{e} AS {n}" for n, e in EXPRESSIONS_DERIVEES.items())
    return f"SELECT *, {deriv} FROM (SELECT *, {base} FROM __THIS__)"


def transformateur_preparation() -> SQLTransformer:
    return SQLTransformer(statement=requete_preparation())


def preparer(df: DataFrame) -> DataFrame:
    """Applique la préparation à un DataFrame au format brut (même code que dans le pipeline)."""
    return transformateur_preparation().transform(df)


# ---------------------------------------------------------------------------
# Regroupement des modalités rares (appris sur l'entraînement)
# ---------------------------------------------------------------------------
def modalites_frequentes(df_train: DataFrame, col: str, min_effectif: int) -> list[str]:
    """Modalités dont l'effectif dans l'ENTRAÎNEMENT atteint `min_effectif` (agrégat Spark)."""
    lignes = (df_train.groupBy(col).count().where(F.col("count") >= min_effectif)
              .orderBy(F.desc("count")).select(col).collect())
    return [r[0] for r in lignes]


def transformateur_regroupement(frequentes: dict[str, list[str]]) -> SQLTransformer:
    """SQLTransformer qui crée `<col>_g` : la modalité si elle est fréquente, sinon 'AUTRE'.
    La liste est figée dans la requête : elle est sauvegardée avec le PipelineModel."""
    def liste(vals):
        return ", ".join(litteral_sql(v) for v in vals)
    cols = ", ".join(
        (f"CASE WHEN {c} IN ({liste(v)}) THEN {c} ELSE '{AUTRE}' END AS {c}_g" if v else f"'{AUTRE}' AS {c}_g")
        for c, v in frequentes.items()
    )
    return SQLTransformer(statement=f"SELECT *, {cols} FROM __THIS__")


# ---------------------------------------------------------------------------
# Évaluation : métriques à partir d'une matrice de confusion agrégée par Spark
# ---------------------------------------------------------------------------
def matrice_confusion(df_pred: DataFrame, label: str = "label", pred: str = "prediction"):
    """Matrice de confusion (pandas, petite : k x k) calculée par un seul groupBy Spark."""
    import pandas as pd
    agg = df_pred.groupBy(label, pred).count().toPandas().astype({label: int, pred: int})
    return agg.pivot_table(index=label, columns=pred, values="count", fill_value=0, aggfunc="sum")


def metriques(mc, n_classes: int) -> dict:
    """Exactitude, F1 macro, F1 pondéré, rappel et précision par classe depuis la matrice k x k."""
    import numpy as np
    m = mc.reindex(index=range(n_classes), columns=range(n_classes), fill_value=0).to_numpy(dtype=float)
    vp = np.diag(m)
    support = m.sum(axis=1)
    predits = m.sum(axis=0)
    rappel = np.divide(vp, support, out=np.zeros_like(vp), where=support > 0)
    precision = np.divide(vp, predits, out=np.zeros_like(vp), where=predits > 0)
    f1 = np.divide(2 * precision * rappel, precision + rappel, out=np.zeros_like(vp),
                   where=(precision + rappel) > 0)
    return {"exactitude": vp.sum() / m.sum(), "f1_macro": f1.mean(),
            "f1_pondere": (f1 * support).sum() / support.sum(),
            "rappel": rappel, "precision": precision, "f1": f1, "support": support}
