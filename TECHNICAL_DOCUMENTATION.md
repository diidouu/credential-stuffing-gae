# Documentation Technique Approfondie

**Projet : Détection d'Anomalies Comportementales et Géographiques dans les Logs de Connexion via des Modèles de Graphe Neuronaux (GAE/VGAE)**

**Version : 2.0**

**Date : 20/06/2025**

## Introduction Générale et Contexte Métier

Ce document expose en détail l'architecture technique, les choix méthodologiques et l'implémentation d'un pipeline de détection d'anomalies dans des journaux d'événements de connexion. L'objectif principal est d'identifier des activités malveillantes sophistiquées, telles que le **credential stuffing**, les attaques par **brute force distribuée**, ou les **compromissions de comptes**, qui sont souvent difficiles à détecter avec des méthodes statistiques classiques ou basées sur des règles prédéfinies.

Les approches traditionnelles échouent car elles analysent les événements de manière isolée. Or, la nature des attaques modernes est relationnelle : un même acteur (IP) peut cibler plusieurs comptes, ou un même compte peut être attaqué depuis de multiples origefeuilles (IPs, User-Agents). L'hypothèse fondamentale de ce projet est que la structure des relations entre les entités de connexion (utilisateurs, adresses IP, user-agents, pays) recèle des informations cruciales pour distinguer un comportement légitime d'un comportement anormal.

Pour capturer cette complexité relationnelle, nous avons modélisé le système sous la forme d'un **graphe hétérogène**. Les modèles de **Graphe Auto-Encodeur (GAE)** et **Graphe Auto-Encodeur Variationnel (VGAE)** sont ensuite utilisés pour apprendre une représentation latente (embedding) de la structure "normale" de ce graphe. Les anomalies sont alors identifiées comme des nœuds ou des sous-graphes que le modèle peine à reconstruire, indiquant une déviation par rapport aux motifs appris.

Ce document a pour vocation de servir de référence pour les ingénieurs, les data scientists et les architectes impliqués dans le projet, en fournissant une traçabilité complète des décisions techniques.

---

## 1. Prétraitement et Ingénierie des Caractéristiques (`pretraitement.ipynb`)

Cette étape est fondamentale car la qualité du graphe et, par conséquent, du modèle, dépend directement de la richesse et de la pertinence des données qui lui sont fournies.

### 1.1. Source des Données

-   **Fichier source** : `logs/user_events_with_geoip_25k.csv`
-   **Description** : Ce fichier CSV contient des enregistrements d'événements de connexion, où chaque ligne représente une tentative de connexion (réussie ou échouée) avec des métadonnées associées.
-   **Colonnes clés utilisées** :
    -   `username` : L'identifiant de l'utilisateur.
    -   `ip_address` : L'adresse IP source de la connexion.
    -   `user_agent` : La chaîne identifiant le client (navigateur, application).
    -   `event_type` : Le type d'événement (ex: `login_success`, `login_failure`).
    -   `timestamp` : L'horodatage de l'événement.
    -   `country_name`, `city_name` : Informations de géolocalisation dérivées de l'IP.

### 1.2. Nettoyage et Préparation

Le script effectue les opérations suivantes :
1.  **Chargement des données** avec la bibliothèque `pandas`.
2.  **Gestion des valeurs manquantes** : Les lignes avec des données critiques manquantes (comme `username` ou `ip_address`) sont supprimées pour garantir la cohérence du graphe.
3.  **Conversion des types** : Le `timestamp` est converti en objet `datetime` pour permettre des calculs temporels.

### 1.3. Ingénierie des Caractéristiques (Feature Engineering)

L'objectif est de créer des caractéristiques numériques (`features`) qui décrivent le comportement de chaque entité (nœud) du futur graphe. Ces features sont cruciales pour que le modèle puisse apprendre des motifs pertinents.

#### 1.3.1. Caractéristiques Comportementales

Ces caractéristiques visent à quantifier le comportement de chaque utilisateur et de chaque adresse IP.

1.  **Fréquence de Connexion par Utilisateur (`connection_frequency`)** :
    -   **Objectif** : Détecter les comptes anormalement actifs (potentiellement compromis ou utilisés par des bots).
    -   **Implémentation** : Pour chaque `username`, on compte le nombre total d'événements (`login_success` + `login_failure`).
    -   **Formalisation** : `freq(user_u) = |{e | event_e.username = user_u}|`

2.  **Diversité des User-Agents par Utilisateur (`user_agent_diversity`)** :
    -   **Objectif** : Un utilisateur légitime se connecte généralement depuis un nombre limité d'appareils/navigateurs. Une grande diversité peut indiquer une attaque distribuée ou une compromission.
    -   **Implémentation** : Pour chaque `username`, on compte le nombre d'`user_agent` uniques associés.
    -   **Formalisation** : `diversity(user_u) = |{ua | ∃ e: event_e.username = user_u ∧ event_e.user_agent = ua}|`

3.  **Taux d'Échec par IP (`ip_failure_rate`)** :
    -   **Objectif** : Identifier les adresses IP qui sont à l'origine de nombreuses tentatives de connexion infructueuses, un signe classique d'attaque par brute force ou credential stuffing.
    -   **Implémentation** : Pour chaque `ip_address`, on calcule le ratio : `nombre d'échecs / nombre total de tentatives`.
    -   **Formalisation** : `fail_rate(ip_i) = |{e | event_e.ip = ip_i ∧ type=fail}| / |{e | event_e.ip = ip_i}|`

#### 1.3.2. Agrégation et Sauvegarde

-   Les caractéristiques calculées sont agrégées au niveau de chaque entité unique (utilisateur, IP).
-   Le DataFrame résultant, contenant les données nettoyées et enrichies, est sauvegardé dans `logs/logs_events_clean.csv`. Ce fichier servira d'entrée pour la construction du graphe.

---

## 2. Construction du Graphe Hétérogène (`construcion_graphe.ipynb`)

Cette étape traduit les données tabulaires en une structure de graphe relationnel, que les modèles neuronaux sur graphes peuvent exploiter. Nous utilisons la bibliothèque **PyTorch Geometric (PyG)**, spécialisée dans ce domaine.

### 2.1. Définition Formelle du Graphe

Nous construisons un graphe hétérogène `G = (V, E)` où :
-   `V` est l'ensemble des nœuds, partitionné en plusieurs types : `V = V_user ∪ V_ip ∪ V_ua ∪ V_country`.
-   `E` est l'ensemble des arêtes, représentant les interactions entre les nœuds.

### 2.2. Typologie des Nœuds et de leurs Attributs

Chaque type de nœud encapsule des caractéristiques spécifiques :

-   **Nœuds `user`** :
    -   **Description** : Représentent les comptes utilisateurs.
    -   **Attributs (`x_user`)** : Vecteur de caractéristiques normalisées contenant :
        1.  `connection_frequency`
        2.  `user_agent_diversity`

-   **Nœuds `ip`** :
    -   **Description** : Représentent les adresses IP sources.
    -   **Attributs (`x_ip`)** : Vecteur de caractéristiques normalisées contenant :
        1.  `ip_failure_rate`

-   **Nœuds `user_agent` et `country`** :
    -   **Description** : Représentent les autres entités contextuelles.
    -   **Attributs** : Pour cette version, nous utilisons un vecteur de caractéristiques constant (one-hot encoding implicite via leur index), mais ils pourraient être enrichis (ex: analyse de la chaîne user-agent, données macro-économiques pour les pays).

### 2.3. Création des Arêtes (Relations)

Les arêtes sont créées pour lier les nœuds qui interagissent dans les logs. Nous définissons plusieurs types d'arêtes pour capturer la sémantique des relations :

-   **(`user`, `logs_in_from`, `ip`)** : Lie un utilisateur à une IP depuis laquelle il s'est connecté. C'est la relation centrale.
-   **(`user`, `uses_agent`, `user_agent`)** : Lie un utilisateur à un user-agent.
-   **(`ip`, `located_in`, `country`)** : Lie une IP à son pays d'origine.

Le script itère sur le fichier `logs_events_clean.csv` et crée les arêtes correspondantes. Un mapping (dictionnaire) est maintenu pour associer chaque entité unique (ex: 'user123') à un index entier, requis par PyG.

### 2.4. Encodage et Normalisation des Caractéristiques

-   **Problème** : Les caractéristiques brutes (`connection_frequency`, `ip_failure_rate`, etc.) ont des échelles très différentes. Les réseaux de neurones sont sensibles à cela, ce qui peut biaiser l'apprentissage.
-   **Solution** : Nous appliquons une **normalisation Min-Max** sur chaque caractéristique de nœud.
    -   **Formule** : `x_norm = (x - min(x)) / (max(x) - min(x))`
    -   Cette transformation ramène toutes les valeurs dans l'intervalle `[0, 1]`, assurant que chaque caractéristique contribue de manière équilibrée au processus d'apprentissage.

### 2.5. Sauvegarde de l'Objet Graphe

-   L'objet `Data` de PyG, qui encapsule toute la structure du graphe (nœuds, arêtes, types, attributs), est sérialisé et sauvegardé sur le disque.
-   **Fichiers de sortie** :
    -   `construction/credential_stuffing_graph_v4.pt` : L'objet graphe PyTorch.
    -   `construction/node_mapping_v4.pt` : Le dictionnaire de mapping pour pouvoir retrouver les entités originales à partir de leurs index.

---

## 3. Modélisation et Apprentissage (`gae_model.ipynb`)

C'est le cœur du système de détection. Le modèle GAE/VGAE apprend une représentation compacte et significative de la topologie et des caractéristiques du graphe.

### 3.1. Fondements Théoriques : GAE et VGAE

#### 3.1.1. Le Graphe Auto-Encodeur (GAE)

Le GAE est un modèle d'apprentissage non supervisé pour les graphes. Son architecture est simple et se compose de deux modules :

1.  **Encodeur (Encoder)** : Une GNN (Graph Neural Network), typiquement un **Graphe Convolutional Network (GCN)**, qui prend en entrée la matrice d'adjacence `A` et la matrice des caractéristiques `X` du graphe. Elle produit une représentation latente `Z` (embedding) pour chaque nœud.
    -   **Formule d'une couche GCN** : `H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))`
        -   `Ã = A + I` (matrice d'adjacence avec self-loops)
        -   `D̃` est la matrice de degrés de `Ã`.
        -   `W^(l)` est la matrice des poids apprenables de la couche `l`.
        -   `σ` est une fonction d'activation non linéaire (ex: ReLU).
    -   Notre encodeur est composé de deux couches GCN. La sortie `Z` est le résultat de la deuxième couche.

2.  **Décodeur (Decoder)** : Une fonction simple qui tente de reconstruire la matrice d'adjacence `A` à partir des embeddings `Z`.
    -   **Mécanisme** : Le **produit scalaire (inner product)**. La probabilité d'une arête entre les nœuds `i` et `j` est modélisée par le produit de leurs embeddings.
    -   **Formule** : `p(A'_ij = 1 | z_i, z_j) = σ(z_i^T z_j)`, où `σ` est la fonction sigmoïde pour ramener le score entre 0 et 1.

3.  **Fonction de Perte (Loss Function)** : L'objectif est de minimiser l'erreur de reconstruction. On utilise une **entropie croisée binaire (Binary Cross-Entropy)** entre la matrice d'adjacence originale `A` et la matrice reconstruite `A'`.
    -   **Formule** : `L = - Σ [A_ij * log(A'_ij) + (1 - A_ij) * log(1 - A'_ij)]`

#### 3.1.2. Le Graphe Auto-Encodeur Variationnel (VGAE)

Le VGAE est une version générative et probabiliste du GAE. Il n'apprend pas seulement un vecteur d'embedding `z_i` pour chaque nœud, mais une **distribution de probabilité** (généralement une Gaussienne) sur l'espace latent, caractérisée par une moyenne `μ_i` et une variance `σ_i^2`.

-   **Encodeur** : Le GCN produit deux vecteurs pour chaque nœud : `μ` et `log(σ)`.
-   **Échantillonnage (Reparameterization Trick)** : On échantillonne l'embedding `Z` à partir de cette distribution : `Z = μ + ε * σ`, où `ε ~ N(0, 1)`.
-   **Décodeur** : Identique au GAE (produit scalaire).
-   **Fonction de Perte (ELBO)** : La perte combine l'erreur de reconstruction avec un terme de régularisation, la **divergence de Kullback-Leibler (KL)**.
    -   **Formule** : `L = L_reconstruction + β * D_KL(q(Z|X,A) || p(Z))`
        -   `L_reconstruction` est la même que pour le GAE.
        -   `D_KL` mesure l'écart entre la distribution apprise `q` et une distribution a priori `p` (généralement une Gaussienne standard `N(0, I)`). Ce terme force l'espace latent à être bien structuré et continu, évitant l'overfitting.
        -   `β` est un hyperparamètre pour pondérer le terme de régularisation.

### 3.2. Implémentation du Modèle

-   **Chargement du graphe** : Le script charge `credential_stuffing_graph_v4.pt`.
-   **Définition de l'architecture** : Des classes PyTorch `GAE` et `VGAE` sont définies, héritant des modèles de base de PyG pour plus de simplicité. L'encodeur est un `GCN` à deux couches.
-   **Boucle d'entraînement** :
    -   **Optimiseur** : `Adam` est utilisé pour sa robustesse et son efficacité.
    -   **Époques** : Le modèle est entraîné pendant un nombre fixe d'époques (ex: 200).
    -   **Processus** : À chaque époque, le graphe entier est passé à travers le modèle (full-batch training). La perte est calculée, la rétropropagation est effectuée, et les poids sont mis à jour. La perte est affichée périodiquement pour suivre la convergence.

### 3.3. Calcul du Score d'Anomalie

-   **Principe** : Une fois le modèle entraîné sur l'ensemble du graphe, il a appris ce qu'est une structure "normale". Les nœuds qui participent à des relations inhabituelles (ex: un utilisateur avec une fréquence de connexion et une diversité d'UA très élevées, se connectant depuis une IP avec un fort taux d'échec) seront mal reconstruits.
-   **Calcul** :
    1.  Le modèle entraîné est utilisé pour générer les embeddings finaux `Z`.
    2.  Le décodeur reconstruit la matrice d'adjacence `A'`.
    3.  Pour chaque nœud `i`, on calcule son **erreur de reconstruction** en comparant ses arêtes originales (`A_i*`) avec ses arêtes reconstruites (`A'_i*`). Ceci est fait en calculant la perte de reconstruction (BCE) au niveau de chaque nœud.
    -   **Un score d'anomalie élevé pour un nœud signifie que le modèle a eu du mal à prédire correctement ses connexions.**

---

## 4. Analyse des Résultats et Validation (`results_v4_behavioral/`)

### 4.1. Identification des Anomalies Principales

-   Les scores d'anomalie de tous les nœuds sont calculés et triés par ordre décroissant.
-   Le script se concentre sur les nœuds de type `user` et `ip` car ils sont les acteurs principaux des scénarios d'attaque.
-   Les `N` nœuds avec les scores les plus élevés sont considérés comme les anomalies les plus probables.
-   Le résultat est sauvegardé dans `results_v4_behavioral/top_anomalies.csv`. Ce fichier contient l'identifiant du nœud, son type, son score d'anomalie, et ses caractéristiques originales pour faciliter l'interprétation.

### 4.2. Validation et Interprétabilité

C'est une étape cruciale pour valider la pertinence du modèle. Il ne suffit pas de produire une liste de scores ; il faut comprendre **pourquoi** le modèle a signalé un nœud comme anormal.

-   **Exemple d'analyse (cas d'un utilisateur)** :
    -   **Score élevé** : L'utilisateur `user_X` a un score d'anomalie de 0.95.
    -   **Analyse des caractéristiques** : En regardant `top_anomalies.csv`, on voit que `user_X` a une `connection_frequency` de 50 (très au-dessus de la moyenne) et une `user_agent_diversity` de 15 (également très élevée).
    -   **Analyse relationnelle** : En explorant le graphe (via des requêtes sur les logs ou des visualisations), on découvre que ce `user_X` est connecté depuis 12 adresses IP différentes, dont plusieurs ont un `ip_failure_rate` élevé.
    -   **Conclusion** : La combinaison de ces facteurs (comportement intrinsèque anormal + relations avec des entités suspectes) justifie pleinement le score d'anomalie élevé. Ce profil est hautement compatible avec un compte compromis utilisé dans le cadre d'une attaque distribuée.

### 4.3. Visualisation (TSNE)

-   Pour aider à l'interprétation globale, une visualisation des embeddings de nœuds via l'algorithme **t-SNE** est réalisée.
-   **Objectif** : Projeter les embeddings de haute dimension (ex: 16D) dans un espace 2D visualisable.
-   **Interprétation** : On s'attend à ce que les nœuds anormaux (ceux avec des scores élevés) se situent dans des régions isolées ou à la périphérie du cluster principal des nœuds "normaux", confirmant que le modèle les a bien séparés dans l'espace latent.

---

## 5. Architecture, Reproductibilité et Évolutions

### 5.1. Dépendances et Environnement

-   Le projet est développé en Python 3.
-   Les dépendances principales sont listées dans `requirements.txt`. Elles incluent :
    -   `torch` et `torch_geometric` (PyG) : Pour la modélisation sur graphes.
    -   `pandas`, `numpy`, `scikit-learn` : Pour la manipulation et la préparation des données.
    -   `matplotlib`, `seaborn` : Pour la visualisation.

### 5.2. Instructions pour la Reproductibilité

Pour exécuter le pipeline de bout en bout :
1.  **Cloner le dépôt** et s'assurer que la structure des dossiers est respectée.
2.  **Installer les dépendances** : `pip install -r requirements.txt`
3.  **Exécuter les notebooks dans l'ordre** :
    1.  `pretraitement.ipynb` : Pour générer `logs_events_clean.csv`.
    2.  `construcion_graphe.ipynb` : Pour générer les fichiers `.pt` du graphe.
    3.  `gae_model.ipynb` : Pour entraîner le modèle et générer les résultats d'anomalie.

### 5.3. Pistes d'Amélioration et Travaux Futurs

-   **Passage à l'échelle (Scalability)** : Pour des volumes de données beaucoup plus importants, l'entraînement en full-batch devient impossible. Il faudrait implémenter des techniques d'échantillonnage de voisins (`NeighborSampler` de PyG) pour permettre un entraînement par mini-batch.
-   **Modèles plus complexes** : Explorer des encodeurs plus puissants comme **Graph Attention Networks (GAT)**, qui peuvent pondérer l'importance des voisins, ou des modèles pour graphes hétérogènes comme **RGCN** ou **HGT**.
-   **Aspect temporel** : Le graphe actuel est statique. Intégrer la dimension temporelle en utilisant des modèles de **graphes dynamiques** (Dynamic GNNs) pourrait permettre de capturer l'évolution des comportements et de détecter des anomalies temporelles (ex: un changement soudain de comportement).
-   **Enrichissement des caractéristiques** : Intégrer davantage de données contextuelles (ex: réputation des IP, informations sur les ASN, analyse plus fine des user-agents) pour améliorer la performance du modèle.
