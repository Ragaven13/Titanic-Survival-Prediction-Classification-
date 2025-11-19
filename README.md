![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-brightgreen?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-blue)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)


🚢 Titanic Survival Prediction using Explainable Machine Learning
    
    An end-to-end machine learning project predicting Titanic passenger survival using classical ML models and SHAP-based model interpretability. This project demonstrates real-world feature engineering, model comparison, tuning, and explainability techniques.

🧠 Project Overview

    The goal of this project is to build predictive models that determine whether a Titanic passenger survived based on demographic, socio-economic, and ticket-related features.
    
    This project includes:
    
    Data cleaning & preprocessing
    
    Feature engineering (Title, Family Size, Deck, etc.)
    
    ML model training (LR, Decision Tree, Random Forest)
    
    Hyperparameter tuning
    
    SHAP explainability (global + local)
    
    Performance visualization

📂 Dataset

    Source: Kaggle – Titanic: Machine Learning from Disaster
    https://www.kaggle.com/c/titanic
    
    Files used:
    
    train.csv
    
    test.csv (optional for inference)

🏗️ Feature Engineering
    Feature	Description
    Title	Extracted from passenger name (Mr, Miss, Mrs, Master, Royalty, etc.)
    FamilySize	SibSp + Parch + 1
    Deck	First letter of Cabin or “Unknown” if missing
    IsAlone	Indicator for passengers traveling alone
    Age/Fare Imputation	Missing ages & fares filled using median
    Categorical Encoding	One-hot encoded categorical columns
    
    These features significantly improved model performance compared to raw dataset.

⚙️ Models Implemented

    Logistic Regression
    
    Decision Tree Classifier
    
    Random Forest Classifier
    
    GridSearchCV for hyperparameter tuning
    
    SHAP explainability tools

📊 Model Performance
    Model	Accuracy
    Logistic Regression	0.8212
    Decision Tree	0.8212
    Random Forest	0.8324
    📌 Interpretation
    
    Logistic Regression and Decision Tree perform equally well at ~82.1%.
    
    Random Forest performs best (83.24%), capturing non-linear relationships and feature interactions.
    
    Shows that feature engineering was effective and improved linear + tree models.

🧠 SHAP Model Explainability

    SHAP (SHapley Additive exPlanations) is used to interpret:
    
    Global feature importance (Sex, Pclass, Fare, Title, Family Size, etc.)
    
    Local explanations for individual passengers
    
    How each feature pushes a prediction toward survival or non-survival
    
    Global Importance:
    shap.summary_plot(shap_values[1], X_train)
    
    Local Passenger Explanation:
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][i],
        X_train.iloc[i],
        matplotlib=True
    )
    
    
    SHAP ensures transparency and helps identify bias (e.g., gender or class influence).



🚀 How to Run

    Install dependencies:
    
    pip install -r requirements.txt
    
    
    Run the training script:
    
    python train.py
    
    
    Optional Streamlit dashboard:
    
    streamlit run shap_dashboard.py

🔮 Future Improvements

    Add XGBoost / LightGBM for higher accuracy
    
    Perform robust cross-validation
    
    Add ROC-AUC curves and precision-recall plots
    
    Deploy SHAP dashboard using Streamlit Cloud
    
    Add bias/fairness analysis

🏁 Conclusion

    This project showcases:
    
    Practical feature engineering
    
    Classical ML modeling
    
    Model comparison
    
    Transparent explainability using SHAP

Strong performance (up to 83.24% accuracy)

Perfect for ML Engineer, Data Scientist, or AI portfolio demonstration.
