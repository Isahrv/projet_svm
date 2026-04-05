# Rapport de Projet : Prédiction de la Performance Étudiante

## 1. Introduction et Objectifs

L'objectif de ce projet est de prédire la note d'examen d'élèves (Exam_Score) à partir de 19 variables explicatives (facteurs socio-démographiques, habitudes d'études, environnement scolaire).

Le jeu de données utilisé, StudentPerformanceFactors, provient de Kaggle et est un jeu de données fictif qui contient 6607 observations, garantissant ainsi une base statistique solide pour l'entraînement de modèles de Machine Learning.

Afin de répondre à la problématique, notre démarche suit un pipeline analytique complet :

- Séparation initiale des données (Train/Test Split).
- Analyse Exploratoire (EDA) et nettoyage.
- Gestion des valeurs extrêmes et manquantes.
- Prétraitement et transformation des variables.
- Entraînement, optimisation et comparaison de plusieurs modèles de Machine Learning.
- Explicabilité globale et locale du modèle retenu.

---

## 2. Préparation des données et Analyse Exploratoire (EDA)

### 2.1. Train/Test Split

La première action réalisée dans notre code a été de séparer nos données en un jeu d'entraînement (X_train, y_train, représentant 80% du jeu de données de base) et un jeu de test (X_test, y_test, 20%) avant toute exploration approfondie ou imputation.

En effet, séparer les données en amont est essentiel pour éviter toute fuite de données (Data Leakage). L'analyse exploratoire, le calcul des quantiles pour le traitement des valeurs extrêmes, ainsi que les règles d'imputation doivent être strictement appris sur le jeu d'entraînement. Le jeu de test doit rester totalement invisible au modèle jusqu'à l'évaluation finale pour refléter correctement sa capacité de généralisation sur de nouvelles données.

Nous avons également vérifié que la distribution de la variable cible Exam_Score était équilibrée et similaire entre les jeux de test et d'entraînement, permettant de s’assurer que le jeu de test peut faire une bonne représentation du jeu d’entraînement.

---

### 2.2. Exploration et Traitement des Valeurs Extrêmes

Après cela, nous avons effectué une analyse univariée, nous permettant d’observer la distribution et les caractéristiques de nos variables.

Nous avons alors constaté que la majorité des variables quantitatives suivent une distribution proche de la normale. Néanmoins, quelques variables comme Hours_Studied, Previous_Scores, Tutoring_Sessions et Physical_Activity présentaient des outliers.

Ainsi, plutôt que de supprimer ces lignes et perdre de la donnée, ce qui réduirait la taille de notre dataset, d’autant plus que nous ne pourrions pas effectuer cette opération sur le jeu de test, nous avons opté pour une winsorisation aux quantiles 1% et 99%. Le traitement de ces valeurs est très important, car les modèles que nous allons comparer (notamment la Régression Linéaire et le SVR) sont très sensibles aux valeurs extrêmes qui peuvent introduire des biais dans les résultats.

La variable cible (Exam_Score) n'a cependant pas été modifiée, afin de ne pas dénaturer ce que l’on cherche à prédire, puisqu’un score d'examen extrême est une réalité terrain que le modèle doit apprendre à gérer.

Une analyse des corrélations a également été réalisée afin d’identifier les relations potentielles entre les variables explicatives, ainsi qu’entre ces variables et la variable cible Exam_Score. Cette étape est essentielle pour mieux comprendre la structure des données et orienter les choix de modélisation.

Dans un premier temps, pour les variables numériques, nous avons utilisé la corrélation de Spearman. Contrairement à la corrélation de Pearson, qui mesure uniquement les relations linéaires, la corrélation de Spearman repose sur les rangs des observations et permet ainsi de détecter des relations monotones, qu’elles soient linéaires ou non

Dans un second temps, pour les variables qualitatives, nous avons utilisé la statistique de Cramer’s V. Cette mesure permet d’évaluer la force de l’association entre deux variables catégorielles, en s’appuyant sur le test du Chi², en obtenant une valeur normalisée comprise entre 0 (absence de relation) et 1 (relation forte), ce qui facilite l’interprétation.

Cependant, les résultats obtenus montrent que, dans l’ensemble, les corrélations entre variables sont très faibles, qu’il s’agisse des variables numériques ou catégorielles. Aucune relation forte ne se dégage clairement. Cela s’explique probablement par le caractère fictif du jeu de données, qui ne reproduit pas fidèlement les dépendances structurelles observées dans des données réelles. Cette faible corrélation limite donc le réalisme du dataset et pourrait alors réduire la capacité des modèles à capter des relations significatives.

Enfin, nous avons étudié les corrélations entre les variables explicatives et la variable cible Exam_Score.

Pour les variables numériques, les corrélations les plus marquées (bien que restant modérées) sont observées pour Attendance et Hours_Studied, toutes deux positivement corrélées avec la note d’examen. Cela suggère que plus un élève est assidu et consacre du temps à l’étude, meilleures sont ses performances, ce qui est cohérent avec les attentes métier.

Concernant les variables catégorielles, même si les associations restent globalement faibles, certaines variables présentent des écarts de score plus marqués entre leurs modalités. C’est notamment le cas de Access_to_Resources, Parental_Involvement, Family_Income et Learning_Disabilities. Ces variables semblent donc introduire des différences notables de performance entre groupes d’élèves, ce qui indique qu’elles pourraient jouer un rôle explicatif important dans les modèles prédictifs.

Globalement, bien que les corrélations observées soient globalement faibles, certaines variables clés émergent comme potentiellement influentes, justifiant leur intégration dans les étapes de modélisation ultérieures.

### 2.3. Prétraitement (Imputation, Encodage et Mise à l'échelle)

Les différents traitements effectués avant modélisation ont tous été faits à partir du jeu train, puis appliqué aux jeux train et test, pour éviter tout data leakage.

Nous avons ensuite repéré des valeurs manquantes dans les colonnes Teacher_Quality, Parental_Education_Level et Distance_from_Home. Ces variables étant catégorielles, leur pourcentage de valeurs manquantes étant très faibles, et l’analyse descriptive ne nous rapportant pas beaucoup d’informations nous permettant de déduire des façons d’imputer ces valeurs manquantes, nous avons décidé de les remplacer par la valeur la plus fréquente pour chacune de ces variables.

Concernant le feature engineering, nous avons créé une nouvelle variable appelée Parent_Context, qui combine le niveau d’éducation des parents et le revenu familial. L’objectif est de capturer un effet d’interaction entre ces deux variables, car leur combinaison peut être plus explicative de l’environnement socio-économique de l’élève que chacune prise séparément. Cela permet potentiellement d’améliorer la capacité prédictive du modèle en introduisant une information plus riche. Nous avons ensuite observé la distribution de cette variable, et constaté que 2 modalités étaient très peu représentées (<6%). Afin d'améliorer la pertinence de la variable pour notre modèle, nous avons décidé de regrouper ces 2 catégories en une seule catégorie "Autres".

Puis, avant la modélisation, certains traitements ont dû être effectués. Notamment l’encodage des variables catégorielles.

Pour les variables catégorielles à faible cardinalité (ex : Genre, Activités extrascolaires), nous avons utilisé le OneHotEncoder. Pour les variables à plus forte cardinalité, nous avons utilisé un TargetEncoder. Cette méthode consiste à remplacer chaque catégorie par la moyenne de la variable cible associée à cette catégorie. Cela permet de réduire fortement la dimensionnalité tout en conservant une information statistique pertinente, contrairement au OneHotEncoder qui aurait généré un grand nombre de variables peu informatives et potentiellement du bruit.

Enfin, nous avons standardisé nos données, pour les variables numériques (sans prendre en compte les variables ayant été encodées qui sont donc déjà standardisées). Pour ce faire, nous avons appliqué la méthode du StandardScaler . Cela est notamment utile pour l'algorithme du SVR qui base son optimisation sur le calcul de distances géométriques dans l'espace des features. Ainsi, sans standardisation, une variable avec une grande échelle (ex: Previous_Scores) écraserait complètement l'impact des variables à petite échelle (ex: Tutoring_Sessions). De plus, cela garantit une convergence plus rapide pour le modèle linéaire.

---

## 3. Modélisation et Comparaison des Modèles

Par la suite, pour répondre à notre objectif de prédiction, nous avons implémenté et comparé différentes familles d'algorithmes. Cette diversité permet de confronter différentes hypothèses mathématiques sur la structure des données.

### 1. Régression Linéaire Multiple :

Ce modèle pose l'hypothèse d'une relation strictement linéaire entre nos facteurs et la note d'examen. Cela nous sert alors de modèle de référence, ce qui sous-entend que si les modèles complexes ne font pas mieux, c'est que la relation sous-jacente est simple.

### 2. Random Forest Regressor :

Ensuite, nous avons testé un modèle de Random Forest, modèle ensembliste basé sur la méthode du Bagging. Son grand atout est sa capacité à modéliser des interactions nonlinéaires complexes sans nécessiter de standardisation des données, même si celle-ci n’impacte généralement pas ses performances. Il est également très résilient face au surapprentissage (overfitting) grâce à la construction de multiples arbres de décision indépendants.

### 3. AdaBoost Regressor :

Nous avons également implémenté un modèle AdaBoost, qui repose sur une approche de boosting. Contrairement au bagging , le boosting construit les modèles de manière séquentielle, chaque nouveau modèle cherchant à corriger les erreurs du précédent. AdaBoost accorde ainsi plus de poids aux observations mal prédites au fil des itérations. Ce type de modèle est particulièrement efficace pour capturer des relations complexes, mais il peut être sensible au bruit et aux valeurs aberrantes, ce qui justifie d’autant plus le travail préalable de winsorisation.

### 4. XGBoost Regressor :

Puis, nous avons testé le modèle XGBoost, une implémentation optimisée du gradient boosting. Il repose sur une descente de gradient appliquée à des arbres de décision, tout en intégrant des mécanismes de régularisation (L1 et L2) afin de limiter le surapprentissage. XGBoost est réputé pour ses performances élevées sur les données tabulaires, notamment grâce à sa capacité à gérer efficacement les interactions complexes entre variables et à optimiser automatiquement la structure des arbres.

### 5. Support Vector Regressor (SVR) :

Enfin, le Support Vector Regressor, faisant partie de la famille des SVM, permet de projeter les données dans un espace de plus grande dimension où la relation devient séparable/linéaire. Il vise à minimiser l'erreur tout en tolérant un certain écart grâce à sa marge. C'est un modèle très puissant sur les datasets de taille moyenne comme le nôtre.

---

### Optimisation et Résultats :

Afin d’obtenir les meilleurs modèles possibles, pour chacun des modèles testés les hyperparamètres ont été optimisés via validation croisée, en commençant par une recherche large à l’aide d’un RandomizedSearchCV, puis d’une recherche plus précise auprès des meilleurs hyperparamètres ressortant, à partir d’un GridSearchCV, en cherchant le modèle avec le moins d’erreurs, minimisant le RMSE. Cela permet d'ajuster le compromis biais-variance (par exemple, gérer la profondeur des arbres pour le Random Forest ou le paramètre de régularisation pour le SVR).

A l’issue de la comparaison des résultats, c’est le modèle SVR qui s’est avéré être le plus performant, minimisant le MSE, MAE, et RMSE, tout en maximisant le R2.

Le meilleur modèle obtenu est donc un SVR avec les hyperparamètres suivants :
{'C': 700, 'epsilon': 0.8, 'kernel': 'linear'}


C (paramètre de régularisation = 700) :

Ce paramètre contrôle le compromis biais-variance. Une valeur élevée de C signifie que le modèle pénalise fortement les erreurs, cherchant donc à s’ajuster au plus près des données d’entraînement. Cela réduit le biais mais augmente le risque de surapprentissage. Ici, la valeur élevée indique que les données présentent une structure suffisamment stable pour supporter un ajustement précis.

epsilon = 0.8 :

Ce paramètre définit la largeur de la “zone de tolérance” autour de la prédiction dans laquelle les erreurs ne sont pas pénalisées. Une valeur relativement élevée signifie que le modèle tolère de petites erreurs sans chercher à les corriger, ce qui améliore la robustesse face au bruit et évite un ajustement excessif.

kernel = linear :

Le choix d’un noyau linéaire indique que la relation entre les variables explicatives et la cible est globalement linéaire. Cela confirme les observations faites précédemment : malgré la complexité potentielle des données, une structure linéaire semble suffisante pour expliquer la majorité de la variance.

Cependant, nous pouvons tout de même noter qu’il ne dépasse le modèle simple de régression linéaire que de peu. Dans notre cas, nous choisirons donc de suivre les chiffres et d’interpréter le modèle SVR dans la suite de notre projet. Néanmoins, dans un cas concret, l’application d’un modèle linéaire pourrait être plus pratique car plus simple à mettre en place et à interpréter qu’un modèle complexe comme un SVR qui n’apporte finalement pas beaucoup d’informations supplémentaires.

---

## 4. Explicabilité et Interprétabilité du Modèle (SVR)

En contexte académique comme en entreprise, l'adoption d'un modèle complexe comme un SVR est souvent freinée par son aspect "boîte noire". Il peut en effet être plus compliqué de les interpréter correctement. Nous avons donc appliqué des méthodes d'explicabilité globales et locales.

### 4.1. Interprétabilité Globale

Interprétation des coefficients :

Pour commencer, l’analyse des coefficients permet d’évaluer l’influence marginale de chaque variable sur la note d’examen (Exam_Score), toutes choses égales par ailleurs. Un coefficient positif indique alors qu’une augmentation (ou la présence) de la variable est associée à une hausse de la note, tandis qu’un coefficient négatif indique l’effet inverse.

Globalement, les résultats mettent en évidence trois grands axes explicatifs de la performance étudiante :

1. Les efforts académiques et l’engagement personnel comme facteurs dominants

Les variables les plus influentes sont directement liées au comportement scolaire de l’élève. L’assiduité (Attendance, coefficient ≈ +2.30) apparaît comme le facteur le plus déterminant, suivie du temps d’étude (Hours_Studied, ≈ +1.83). Cela confirme que la présence en cours et le travail personnel sont les leviers principaux de la réussite académique.

Les performances passées (Previous_Scores, ≈ +0.71) jouent également un rôle important, traduisant une forme d’inertie académique : les bons élèves ont tendance à le rester.

Enfin, les sessions de tutorat (Tutoring_Sessions, ≈ +0.62) ont un impact positif, suggérant que l’accompagnement pédagogique contribue à améliorer les résultats.

2. L’environnement éducatif et socio-économique comme facteurs structurants

Un second groupe de variables met en évidence l’importance du contexte dans lequel évolue l’élève. Un fort engagement parental (Parental_Involvement_High, ≈ +1.00), un bon accès aux ressources (Access_to_Resources_High, ≈ +1.00), ainsi qu’un revenu familial élevé (Family_Income_High, ≈ +0.53) sont associés à de meilleures performances.

De même, un environnement scolaire favorable, caractérisé par une bonne qualité des enseignants (Teacher_Quality_High, ≈ +0.54) et une influence positive des pairs (Peer_Influence_Positive, ≈ +0.53), contribue significativement à la réussite.

L’accès à Internet (Internet_Access_Yes, ≈ +0.49) et un niveau d’éducation parental élevé (Postgraduate, ≈ +0.43) renforcent également cette tendance. Ces résultats traduisent le rôle clé du capital social et culturel dans la réussite scolaire.

3. Les effets négatifs associés aux contextes défavorables

À l’inverse, les coefficients négatifs reflètent des situations moins propices à la réussite. Un faible accès aux ressources (Access_to_Resources_Low, ≈ -0.98) et un faible engagement parental (Parental_Involvement_Low, ≈ -0.97) figurent parmi les facteurs les plus pénalisants.

Des variables telles qu’un revenu faible (Family_Income_Low, ≈ -0.49), une mauvaise qualité d’enseignement (Teacher_Quality_Low, ≈ -0.50), une influence négative des pairs (Peer_Influence_Negative, ≈ -0.51) ou encore l’absence d’accès à Internet (Internet_Access_No, ≈ -0.49) contribuent également à une baisse des performances.

La présence de troubles d’apprentissage (Learning_Disabilities_Yes, ≈ -0.47) est logiquement associée à une diminution de la note, tandis qu’un éloignement important du domicile (Distance_from_Home_Far, ≈ -0.50) peut refléter des contraintes logistiques impactant la réussite.

4. Variables faiblement explicatives

Certaines variables présentent des coefficients très proches de zéro, comme le genre, le type d’établissement ou encore le nombre d’heures de sommeil. Nous retrouvons parmis celles-ci également notre variable créee, “Parent_Context”. Cela suggère qu’elles ont un impact marginal sur la performance dans ce jeu de données, ou que leur effet est déjà capturé indirectement par d’autres variables.

Dans l’ensemble, le modèle met en évidence que la réussite académique repose principalement sur un triptyque : l’investissement personnel de l’élève, la qualité de son environnement éducatif et le contexte socio-économique. Les variables comportementales apparaissent comme les plus déterminantes, ce qui souligne que, même si le contexte joue un rôle important, les efforts individuels restent le levier principal de performance.

Nous pouvons également noter que ces résultats sont similaires aux coefficients déterminés par le modèle simple de régression linéaire.

Distribution des résidus : Ensuite, l’observation de la distribution des résidus nous montre que malgré la présence de quelques valeurs dispersées, la majorité des résidus sont répartis de manière homogène autour de 0. Cela indique que le modèle ne présente pas de biais systématique (ni sous-estimation ni surestimation globale).

Importance des Variables selon les coefficients : Globalement, nos résultats montrent que des variables telles que l'assiduité (Attendance), les notes précédentes (Previous_Scores) et le temps d'étude (Hours_Studied) sont les moteurs principaux de la performance à l'examen, ayant un coefficient positif sur notre variable cible.

Partial Dependence Plots (PDP) : Ensuite, l’analyse des graphiques de Partial Dependence Plots nous ont permis de vérifier la relation marginale des features. Par exemple, la note d'examen augmente de manière quasi monotone avec Attendance, prouvant que le modèle a bien capté la logique métier : un élève davantage présent a de meilleures notes qu’un élève moins présent. Nous pouvons noter que le modèle SVR sélectionné étant linéaire, il n’est pas nécessaire d’observer les ALE ou ICE, puisque cela ne nous apportera pas d’informations supplémentaires.

### 4.2. Interprétabilité Locale : LIME & SHAP

Pour comprendre finement la prise de décision sur un cas spécifique, nous avons extrait les prédictions de différents individu ; l'individu n°505 via LIME (Local Interpretable Model-agnostic Explanations) et l’individu n°86 via SHAP (SHapley Additive exPlanations).

Résultats de l'explication locale (LIME) :

Premièrement, LIME permet de générer un modèle linéaire de substitution, interprétable autour du voisinage immédiat de notre individu. Pour l'étudiant 505, la prédiction est influencée par :

- Un effet fortement positif de son excellente assiduité (Attendance > 0.88 génère +4.08 points).
- Un effet positif de ses bons résultats passés (Previous_Scores > 0.90 génère +1.25 points).

Cependant, LIME nous montre que pour cet étudiant précis, le nombre d'heures étudiées (Hours_Studied <= -0.67) et les sessions de tutorat (Tutoring_Sessions <= -0.41) tirent sa note vers le bas (respectivement -3.33 et -1.04).

Ces résultats locaux peuvent alors sembler contre-intuitifs par rapport à la tendance globale (où étudier plus aide généralement à avoir de meilleures notes). En fait, le modèle SVR a identifié un comportement complexe propre au profil de cet étudiant. On peut par exemple formuler l'hypothèse suivante : un élève ayant déjà de très bonnes notes précédentes et une forte présence en classe, mais qui a soudainement besoin de tutorat ou diminue radicalement ses heures de travail autonome, pourrait subir une baisse de performance. Le modèle ajuste donc localement son score à la baisse.

SHAP values :

Les Waterfall et beeswarm plots fait à partir de SHAP confirme les constats globaux fait précédemment. En partant de la valeur moyenne de tous les élèves (Expected Value), le graphique ajoute et soustrait les valeurs de Shapley exactes de chaque caractéristique de l'étudiant 86 pour aboutir à sa prédiction finale exacte. Cela permet une décomposition parfaite de la prédiction sans biais d'approximation.

interprétation

---

## 5. Conclusion

Pour conclure, ce projet avait pour objectif de prédire la performance académique d’étudiants à partir de variables socio-démographiques et comportementales, en mobilisant un pipeline complet de data science.

Les différentes étapes de la préparation des données à l’explicabilité du modèle ont permis de mettre en évidence plusieurs points clés. Tout d’abord, le prétraitement des données (gestion des valeurs manquantes, traitement des outliers, encodage et standardisation) joue un rôle déterminant dans la performance des modèles. Ensuite, la comparaison de plusieurs algorithmes a montré que, malgré la complexité des méthodes ensemblistes ou du boosting, un modèle relativement simple comme le SVR linéaire (ou même la régression linéaire) peut être tout aussi performant ou même davantage performant qu’un modèle plus complexe.

Dans notre cas, cela s’explique notamment par la structure des données, qui présentent peu de corrélations fortes et semblent majoritairement linéaires.

Puis, l’analyse d’interprétabilité globale a confirmé l’importance de variables clés telles que l’assiduité, les notes passées et le temps d’étude, ce qui est cohérent avec la logique métier. Les méthodes locales ont également permis d’aller plus loin, en expliquant des cas individuels, mettant en évidence des comportements spécifiques que les analyses globales ne permettent pas de détecter.

Enfin, ce projet illustre un point essentiel en data science : le modèle le plus performant n’est pas toujours le plus complexe. Dans un cadre opérationnel, un modèle plus simple, interprétable et robuste peut être préférable, notamment pour faciliter la prise de décision et l’appropriation par les parties prenantes.

En perspective, l’amélioration du réalisme du jeu de données, ainsi que l’intégration de variables supplémentaires (psychologiques, pédagogiques ou contextuelles), pourraient permettre d’obtenir des modèles plus performants et plus proches des situations réelles.