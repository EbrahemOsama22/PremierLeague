اتفضل، الـ description بعد الإضافة:

---

## Football Matches Analysis & Prediction 🏆⚽

A comprehensive data science project that analyzes football match data across multiple seasons (2020–2024) using Python. The project covers the full data science pipeline — from raw data to machine learning models.

### 📌 What This Project Does

- **Data Preprocessing** — Handles missing values (e.g., attendance imputation), removes duplicates, and detects/clips outliers using the IQR method.
- **Feature Engineering** — Creates new features like Expected Goals Difference (`XGD`), Goal Difference (`+/-`), and Weekend flag.
- **Exploratory Data Analysis (EDA)** — Visualizes total attendance per season, goals scored/conceded per team, win/loss/draw records, and goal difference trends across seasons.
- **Feature Selection** — Selects the top 80% of most relevant features using `SelectPercentile` with ANOVA F-score (`f_classif`), based on correlation analysis with the target variable.
- **Data Encoding** — Encodes categorical features (team, opponent, formation, venue, etc.) using Label Encoding and One-Hot Encoding for ML readiness.
- **Machine Learning** — Applies multiple classification algorithms to predict match outcomes (Win/Draw/Loss):
  - Logistic Regression
  - **Decision Tree** — tuned with Grid Search over `criterion`, `max_depth`, `min_samples_split`, and `min_samples_leaf`
  - **K-Nearest Neighbors (KNN)** — tuned with Grid Search over `n_neighbors`, `weights`, `metric`, and `leaf_size`
  - Random Forest
  - SVM
  - Gradient Boosting
  - AdaBoost
  - Naive Bayes
  - MLP Neural Network
- **Unseen Data Prediction** — Each model is tested against unseen match samples to validate real-world generalization.

### 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn`

### 📂 Dataset Features

Key columns include: `team`, `opponent`, `result`, `gf`, `ga`, `xg`, `xga`, `poss`, `attendance`, `formation`, `venue`, `season`, and more.

