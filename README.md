# Prédiction de la Performance Étudiante
Par HERVE Isaline et DAERON Djayan - M2 ECAP, IAE Nantes - Année scolaire 2025-2026

## Objectif

Ce projet vise à prédire la **note d'examen des élèves** (*Exam_Score*) à partir de variables liées à leur environnement socio-démographique, leurs habitudes d’étude et leur contexte scolaire.

Le jeu de données utilisé (*StudentPerformanceFactors*, Kaggle) contient **6607 observations** et **19 variables**.

---

## Méthodologie

Le projet suit un pipeline classique de data science :

- Train/Test Split
- Analyse exploratoire (EDA)
- Gestion des valeurs manquantes et des outliers
- Prétraitement (encodage, scaling, feature engineering)
- Modélisation et optimisation
- Comparaison des modèles
- Explicabilité (coefficients, LIME, SHAP)

---

## Modèles testés

- Régression Linéaire  
- Random Forest  
- AdaBoost  
- XGBoost  
- SVR (Support Vector Regressor)  

**Modèle retenu : SVR linéaire**

---

## Résultats

Les variables les plus influentes sont :

- **Attendance (assiduité)**
- **Hours_Studied (temps d’étude)**
- **Previous_Scores (résultats passés)**

Les facteurs environnementaux (ressources, implication parentale, revenu) ont également un impact significatif.

---

## Explicabilité

Le modèle a été interprété avec :

- Coefficients (modèle linéaire)
- Permutation Feature Importance
- LIME (local)
- SHAP (local & global)

---

## Remarque

Le dataset étant fictif, les corrélations sont très faibles. Les résultats restent globalement cohérents mais doivent être interprétés avec prudence.

---

## Python Version

3.11

---

## Reproduire le projet (avec uv)

### 1. Installer uv

```bash
pip install uv
```

### 2. Cloner le projet

```bash
git clone <repo_url>
cd <repo>
```

### 3. Installer les dépendances

```bash
uv sync
```
