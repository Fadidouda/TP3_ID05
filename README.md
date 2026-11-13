# TP3 — Machine Learning : du petit jeu de données au Big Data

Projet de Big Data (école d'ingénieurs, 5e année) en deux étapes : le **même workflow complet de Machine Learning** est appliqué d'abord à un petit jeu de données avec pandas et scikit-learn (**small**), puis à un jeu volumineux avec Spark (**big**).

| Dossier | Étape | Données | Outils | Question |
|---|---|---|---|---|
| [`small_requins/`](small_requins/) | 1 | Attaques de requins (GSAF), 4 087 attaques retenues | pandas, scikit-learn | L'attaque sera-t-elle mortelle ? |
| [`big_feux/`](big_feux/) | 2 | Feux de forêt aux États-Unis (FPA FOD), 1,88 M feux | PySpark 3.5 (Spark SQL, `pyspark.ml`) | Quelle est la cause du feu, dès sa découverte ? |

Les notebooks sont livrés **exécutés** : sorties, graphiques et interprétations sont visibles directement sur GitHub.

---

## Structure

```
TP3/
├── README.md
├── small_requins/
│   ├── etape1_requins.ipynb       # notebook complet (sections 0 à 9)
│   ├── src/preparation.py         # nettoyage réutilisé à l'entraînement et au déploiement
│   ├── models/schema_entree.json  # entrées attendues par le modèle
│   └── data/                      # à remplir (voir « Données »)
└── big_feux/
    ├── etape2_feux.ipynb          # notebook complet (sections 0 à 11)
    ├── src/preparation.py         # préparation en expressions Spark SQL natives
    ├── models/schema_entree.json  # entrées, classes prédites, versions
    ├── results/                   # mesures (ingestion, passage à l'échelle)
    └── data/                      # à remplir (voir « Données »)
```

## Données (non incluses dans le dépôt)

Les fichiers de données ne sont pas versionnés : le SQLite fait 796 Mo, au-delà de la limite de 100 Mo par fichier de GitHub, et les conditions de redistribution sont celles des sources.

- **Étape 1** : télécharger le CSV Kaggle [Shark Attacks](https://www.kaggle.com/datasets/gauravkumar2525/shark-attacks), dérivé du *Global Shark Attack File* (Shark Research Institute), et le placer dans `small_requins/data/global_shark_attacks.csv`.
- **Étape 2** : télécharger la *Fire Program Analysis Fire-Occurrence Database* (US Forest Service), décompresser `FPA_FOD_20170508.sqlite` et le placer dans `big_feux/data/`.
  > Short, K. C. 2017. *Spatial wildfire occurrence data for the United States, 1992-2015 [FPA_FOD_20170508]*. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4

  Le Parquet (`data/fires.parquet`) est créé automatiquement par le notebook à la première exécution.

Les modèles (`.joblib`, `PipelineModel` Spark) sont régénérés par les notebooks.

## Environnement et exécution

Testé sous Windows + WSL2 (Ubuntu 22.04), Python 3.10, OpenJDK 11, portable Intel i5-1235U (12 threads, 8 Go pour WSL).

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib pyarrow psutil pyspark==3.5.9 jupyterlab
```

Lancer Jupyter **depuis le dossier de l'étape** (les chemins sont relatifs à ce dossier), puis *Kernel → Restart Kernel and Run All Cells* :

```bash
cd big_feux && jupyter lab
```

| Notebook | Durée d'exécution complète |
|---|---|
| `small_requins/etape1_requins.ipynb` | ≈ 1 min 40 |
| `big_feux/etape2_feux.ipynb` | ≈ 63 min (Spark `local[*]`, 4 Go pour le driver) |

---

## Étape 1 — small : mortalité des attaques de requins

**Workflow.** Inspection et définition de la cible, exclusion des fuites, période d'étude, EDA, nettoyage testé (`src/preparation.py`), pipeline `ColumnTransformer`, validation croisée stratifiée, réglage, évaluation unique sur le test, seuil métier, importance par permutation, robustesse temporelle, préparation du déploiement.

**Choix principaux.**
- **Période ≥ 1960** : avant, seuls les cas graves étaient recensés (mortalité de 30 à 80 % par décennie, contre moins de 20 % ensuite).
- **Périmètre** : attaques provoquées, non provoquées et sur embarcation. Sont exclus les cas *Invalid* (implication du requin non établie) et *Sea Disaster* (hors cas d'usage baigneurs/surfeurs).
- **Fuites exclues** : aucun texte brut (l'espèce citait les « remains » ou la « death »), seulement des catégories extraites par mots-clés ; noms et lieux précis exclus.
- **Modèle** : random forest, choisi sur la PR AUC en validation croisée.
- **Seuil** : 0,33, qui maximise le F2. En prévention, rater une situation dangereuse coûte plus cher qu'une fausse alerte.

**Résultats sur le test.**

| Modèle | ROC AUC | PR AUC |
|---|---|---|
| **Random forest** | **0,852** | **0,458** |
| Référence « pays seul » | 0,739 | 0,247 |
| Référence naïve | 0,507 | 0,121 |

Au seuil retenu, le modèle repère **81 % des attaques mortelles** (rappel 0,806, précision 0,289). Les situations les plus à risque sont certains pays (La Réunion, Papouasie-Nouvelle-Guinée, Brésil), les requins tigre, blanc et bouledogue, et la plongée ou la baignade plutôt que le surf.

## Étape 2 — big : cause des feux de forêt avec Spark

**Workflow.** Ingestion SQLite → Parquet, cadrage de la cible (11 causes), analyse colonne par colonne des fuites, EDA par agrégations Spark, préparation en SQL Spark testée, **découpage temporel**, pipeline `pyspark.ml` avec poids de classes, deux références, grille de modèles, évaluation unique, fuite mesurée, feux de cause inconnue, **passage à l'échelle Spark contre scikit-learn**, `PipelineModel` sauvegardé.

**Choix principaux.**
- **Ingestion** : lecture pandas par blocs de 200 000 lignes et écriture Parquet avec un schéma explicite (pas de pilote JDBC SQLite disponible). Résultat : **90,7 Mo contre 795,8 Mo** (8,8 fois moins), en 25 s, une seule fois.
- **Cible** : 11 causes connues. « Structure » (0,2 %) est fusionnée avec « Divers ». Les 166 723 feux « Missing/Undefined » sont mis à part pour le cas d'usage.
- **Fuites exclues** : code de cause, taille finale et classe de taille, dates de maîtrise, noms de feux (la cause y figure dans 1,18 % des cas, avec 91 % de justesse pour « LIGHTNING »), liens ICS-209/MTBS, identifiants.
- **Découpage temporel** : entraînement 1992-2009, validation 2010-2012, test 2013-2015. La répartition des causes dérive dans le temps : « Divers » passe de 17 % à 26 %.
- **Préparation** : expressions Spark SQL natives (pas d'UDF), encapsulées dans des `SQLTransformer`. Le modèle déployé applique exactement ce nettoyage à des colonnes brutes.
- **Modèle** : random forest, 30 arbres, profondeur 12, poids de classes. La profondeur 15 provoquait un `OutOfMemoryError` en prédiction avec 4 Go et 12 threads.

**Résultats sur le test 2013-2015 (192 806 feux).**

| Modèle | Exactitude | F1 macro | F1 pondéré |
|---|---|---|---|
| Classe majoritaire | 0,285 | 0,040 | 0,126 |
| Cause dominante de l'État | **0,426** | 0,156 | 0,359 |
| **Random forest** | 0,356 | **0,257** | **0,394** |

- Le F1 macro est **1,65 fois** celui de la meilleure règle simple. L'exactitude est plus basse, car les poids de classes favorisent les causes rares.
- La **foudre** est bien reconnue (F1 0,713). Cigarette, chemin de fer, ligne électrique et équipement restent quasi indistinguables avec les seules informations de découverte.
- **Importance des variables** : géographie 37 %, système de déclaration (propriétaire, organisme, unité) 40 %, saison 17 %. Une partie du modèle apprend donc les habitudes de classement des organismes.
- **Feux de cause inconnue** : prédits bien plus souvent « incendie criminel » (25,6 %) que les feux du test (11,2 %). C'est une hypothèse plausible, pas une preuve.

**Passage à l'échelle (même random forest, même validation).**

| Lignes d'entraînement | Spark | scikit-learn | F1 macro (Spark / sklearn) |
|---|---|---|---|
| 13 099 | 41,7 s | 0,5 s | 0,253 / 0,252 |
| 65 002 | 68,5 s | 0,9 s | 0,273 / 0,270 |
| 260 719 | 119,7 s | 3,9 s | 0,258 / 0,262 |
| 652 550 | 180,7 s | 12,9 s | 0,258 / 0,259 |
| 1 302 822 | 250,9 s | 12,1 s | 0,251 / 0,251 |

Sur une seule machine, **scikit-learn reste environ 20 fois plus rapide**, même avec 1,3 M lignes : les 146 Mo de variables tiennent en mémoire, et Spark (≈ 6 Go pour la JVM) paie un coût fixe. **Le score ne progresse pas avec le volume** : la limite vient des variables disponibles, pas du nombre d'exemples. Spark devient intéressant au-delà de la mémoire d'une machine ou sur un cluster. Ici, il a surtout servi à ingérer, nettoyer et agréger 1,9 M lignes sans les charger dans pandas.

---

## Limites communes

- **Biais de recensement et de déclaration** : cas graves surreprésentés (requins) ; pratiques de classement propres à chaque organisme (feux).
- **Données manquantes non aléatoires** : espèce, heure, cause inconnue.
- **Corrélation, pas causalité** : les modèles désignent des situations ou des pistes à privilégier, pas des causes.
- **Étape 2** : faite sur la 4e édition (1992-2015) fournie, et non sur la 6e (1992-2020). Les résultats sont à confirmer sur cette dernière.
