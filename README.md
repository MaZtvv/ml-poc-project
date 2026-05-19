# PokerMind AI

## Objectif du projet

PokerMind AI est un projet de machine learning appliqué aux historiques de mains de poker. L'objectif principal est d'estimer, à partir d'une main donnée, quel joueur a la plus forte probabilité de gagner. Le projet s'appuie sur des fichiers PHH, un format texte structuré conçu pour décrire des mains de poker de manière standardisée et exploitable par des scripts de parsing.

Le projet avance de manière progressive. La première étape consiste à comprendre la structure réelle des données brutes, puis à construire des tables de features au niveau de la main et du joueur. Une fois cette base en place, l'objectif est de comparer des modèles supervisés simples et interprétables avant d'aller vers des analyses plus fines.

À plus long terme, PokerMind AI vise aussi l'analyse de la qualité des décisions. L'idée sera alors d'aller au-delà du simple résultat final pour étudier les mises, les tailles de stack, les cartes communes, l'équité et les cotes du pot.

## Dataset utilisé

Le projet utilise le **PHH Dataset / Poker Hand Histories**. PHH signifie **Poker Hand History**. Il s'agit d'un format texte structuré, pensé pour représenter des mains de poker de façon lisible et standardisée.

Le sous-ensemble actuellement utilisé pour le travail local est un sous-échantillon **Pluribus No-Limit Texas Hold'em**, situé dans :

```text
data/raw/pluribus/
```

Les fichiers manipulés sont des fichiers :

- `.phh`
- `.phhs`

Il s'agit d'historiques de mains réels au format PHH, et non d'un dataset tabulaire simplifié. La variante actuellement exploitée dans le sous-ensemble de travail est **NT**, c'est-à-dire **No-Limit Texas Hold'em**.

Les données brutes sont placées dans :

```text
data/raw/
```

Les gros dossiers de données brutes ne sont pas poussés sur GitHub. Les données traitées sont stockées dans :

```text
data/processed/
```

## Pourquoi ce dataset est pertinent

Ce dataset est pertinent car il contient les briques nécessaires à un vrai projet d'analyse de poker :

- les joueurs
- les stacks de départ
- les stacks finaux
- les cartes privées si elles sont disponibles
- les cartes du board
- les actions de la main
- le résultat final

Cela permet déjà de construire une première baseline de **win probability modeling**. Plus tard, ce même type de structure pourra aussi servir à construire un dataset d'analyse de qualité de décision.

## Structure des données PHH

Les fichiers PHH contiennent des champs structurés décrivant une main. Quelques éléments importants :

- `variant` : variante de poker jouée
- `players` : liste des joueurs
- `starting_stacks` : stacks de départ
- `finishing_stacks` : stacks finaux
- `actions` : séquence des actions

Quelques codes PHH utiles :

- `d dh` : distribution des cartes privées
- `d db` : distribution des cartes du board
- `pN f` : fold
- `pN cc` : check ou call
- `pN cbr` : complete, bet ou raise

Le format est donc bien adapté à un parsing progressif, puis à la construction de variables métier plus riches.

## Feature engineering actuel

Le projet produit actuellement deux tables traitées.

### 1. `hand_level_features.csv`

Fichier :

```text
data/processed/hand_level_features.csv
```

Cette table contient une ligne par historique de main. Elle sert surtout à valider l'exploitabilité des fichiers PHH avant de construire des datasets plus détaillés.

État actuel du sous-ensemble de travail :

- 300 fichiers PHH Pluribus traités
- 300 lignes au niveau de la main générées
- 300 lignes en variante `NT`
- 300 lignes marquées `usable_for_first_model = True`

### 2. `player_level_features.csv`

Fichier :

```text
data/processed/player_level_features.csv
```

Cette table contient une ligne par joueur dans une main. C'est la première table réellement exploitable pour entraîner des modèles supervisés simples.

Elle contient notamment :

- les informations de main et de joueur
- les stacks de départ et de fin
- un proxy de résultat
- des features simples préflop sur les cartes
- des compteurs d'actions pour un baseline plus large au niveau de la main complète

Variables clés :

- `player_won = 1` si `finishing_stack > starting_stack`, sinon `0`
- `profit = finishing_stack - starting_stack`

Ce choix constitue un **premier proxy de baseline**, pas un modèle parfait de l'issue réelle d'une main.

## Modèles testés

Les modèles actuellement testés dans le projet sont :

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Un clustering exploratoire de type **K-Means** peut aussi être utilisé dans le notebook de modélisation, mais il ne s'agit pas d'un modèle supervisé de prédiction du gagnant.

Le notebook compare deux jeux de variables :

### 1. Features préflop

Ces features n'utilisent que l'information disponible avant que la main ne se développe réellement :

- hole cards
- position ou ordre de siège
- blindes
- starting stack
- nombre de joueurs
- score simple de force préflop

C'est la tâche de prédiction la plus propre d'un point de vue méthodologique.

### 2. Features full-hand

Ces features incluent aussi des informations issues du déroulement de la main :

- folds
- calls
- raises
- board deals
- show / muck

Ce baseline peut avoir de meilleures performances, mais il introduit un **risque de fuite d'information** si on l'interprète comme une prédiction disponible avant les décisions.

## Choix du modèle

Le baseline principal retenu à ce stade est :

- **Logistic Regression sur les features préflop**

Raisons principales :

- bonnes performances de base en accuracy et ROC-AUC sur la tâche préflop
- modèle simple et interprétable
- moins d'overfitting que des modèles d'arbres plus flexibles

La **Logistic Regression full-hand** est conservée comme modèle secondaire de diagnostic, mais pas comme baseline propre principale, car elle exploite des informations qui peuvent être trop proches de l'issue de la main.

## Limites actuelles

Le projet en est encore à une première baseline. Les limites principales sont :

- `player_won` est approché par `finishing_stack > starting_stack`
- aucune estimation d'équité Monte Carlo n'est encore intégrée
- les cotes du pot ne sont pas encore calculées
- il n'existe pas encore de dataset décisionnel street-by-street
- le sous-ensemble Pluribus correspond à un contexte de benchmark IA / poker, pas forcément à des mains d'online poker ordinaires
- il s'agit d'un premier niveau de modélisation, pas d'un système complet d'analyse stratégique

## Prochaines étapes

Les prochaines étapes naturelles du projet sont :

- améliorer le parsing au niveau joueur
- ajouter des features par street
- estimer l'équité
- calculer les cotes du pot
- construire un dataset au niveau de la décision
- créer ensuite une application Streamlit pour visualiser les mains et les prédictions

## Notebooks actuels

### `notebooks/01_data_exploration.ipynb`

Contient :

- inspection des fichiers bruts
- notes sur la grammaire PHH
- EDA globale
- résultats de feature engineering au niveau de la main

### `notebooks/02_baseline_modeling.ipynb`

Contient :

- chargement du dataset joueur
- train/test split
- comparaison de modèles
- comparaison préflop vs full-hand
- importance des variables
- clustering exploratoire si présent

## Structure du projet

```text
PokerMind AI/
├── data/
│   ├── raw/
│   │   └── pluribus/
│   └── processed/
│       ├── hand_level_features.csv
│       └── player_level_features.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_modeling.ipynb
├── src/
│   ├── feature_engineering.py
│   └── player_features.py
├── README.md
└── requirements.txt
```

## Installation et utilisation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Générer la table de features au niveau de la main :

```bash
python src/feature_engineering.py
```

Générer la table de features au niveau du joueur :

```bash
python src/player_features.py
```

Ouvrir le notebook d'exploration :

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Ouvrir le notebook de baseline modeling :

```bash
jupyter notebook notebooks/02_baseline_modeling.ipynb
```

## Avertissement

Ce projet est destiné à des fins éducatives, analytiques et académiques uniquement. Il ne constitue pas un conseil en matière de jeu d'argent et ne doit pas être utilisé pour encourager ou guider des décisions de gambling réel.
