# Kaggle-Notebooks

Welcome to my Kaggle Notebooks repository! This repository contains Jupyter notebooks from various Kaggle competitions, where I explore data, build models, and analyze results.\n
🗂️ Each competition folder also includes a `markdown` subfolder containing the code and output exported as markdown for easy viewing and sharing.

## 📌 Current Competitions

### [CIBMTR - Equity in post-HCT Survival Predictions](https://www.kaggle.com/competitions/cibmtr-equity)
- **Goal:** Develop predictive models for post-hematopoietic cell transplantation (HCT) survival while addressing equity concerns.
- **Notebook:** [`transplanting_data_into_insights.ipynb`](./notebooks/cibmtr_equity/transplanting_data_into_insights.ipynb)
- **Approach:** Data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning models.

### [Playground Series - Predicting Podcast Listening Time](https://www.kaggle.com/competitions/playground-series-s5e4/overview)
- **Goal:** Predict listening times of podcast episodes.
- **Notebook:** [`eda-ensemble-pytorch-xgboost-help-wanted.ipynb`](./notebooks/predict_podcast_listening_time/eda-ensemble-pytorch-xgboost-help-wanted.ipynb)
- **Approach:** Data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning models (XGBoost & PyTorch).

## 📂 Repository Structure
```
📦 kaggle-notebooks
 ┣ 📂 notebooks/              # Kaggle notebooks organized by competition
 ┃ ┣ 📂 cibmtr_equity/        # CIBMTR competition folder
 ┃ ┃ ┣ 📜 transplanting_data_into_insights.ipynb # Main notebook for the competition
 ┃ ┃ ┣ 📜 pip_install_liberies.ipynb  # Notebook handling package installation
 ┃ ┃ ┣ 📜 eefs_concordance_index.ipynb  # Helper notebook containing the competition evaluation metric
 ┃ ┃ ┗  📂 markdown  # Folder containing the competition notebook (.md) and all output figures
 ┃ ┃ ┃ ┣ 📜 transplanting_data_into_insights.md # Main notebook for the competition with rendered figures
 ┃ ┃ ┃ ┗ 📜 xxx.png # Output figures
 ┣ 📜 README.md              # This file
 ┣ 📜 .gitignore             # Ignore unnecessary files
```

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JipWulffele/Kaggle-Notebooks.git
   cd kaggle-notebooks
   ```
2. **Install dependencies:**
   If required, install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebook:**
   Open the notebook using Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. **Using Helper Notebooks:**
   - The main notebooks might rely on helper notebooks for reusable functions.
   - `pip_install_liberies.ipynb` installs necessary packages when Kaggle’s internet is disabled.
   - Ensure helper notebooks are present in `notebooks/xxx/` before running the main notebook.

## 📌 Future Plans
- Add more notebooks from other Kaggle competitions.
- Improve documentation and model explanations.
- Experiment with different machine learning techniques.

## 📜 License
This repository is open-source and available under the MIT License.

---
📩 **Contributions & Feedback:** Feel free to fork the repository, open issues, or suggest improvements!
