# 🩺 Diabetes Risk Prediction

A machine learning project to predict diabetes risk based on patient health data.

## Dataset
- 100,000 patient records
- Source: [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

## Results
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9684 |
| Recall (Diabetes) | 0.75 |
| Precision (Diabetes) | 0.77 |
| Accuracy | 0.96 |

## Project Steps
1. EDA & Data Cleaning
2. Preprocessing (Label Encoding + StandardScaler)
3. Handled class imbalance using SMOTE
4. Trained Logistic Regression & Random Forest
5. Selected Random Forest (best ROC-AUC)
6. Deployed with Streamlit

## Run the App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
```
├── diabetes_prediction_project.ipynb
├── app.py
├── diabetes_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---
*Built by Abanoub George — Final Year Engineering Student*
