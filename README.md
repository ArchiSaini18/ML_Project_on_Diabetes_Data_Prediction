# ML_Project on 🩺 Diabetes Data Prediction 
Developed a machine learning model to predict whether an patient has diabetes based on medical features such as Pregnancies,glucose level, BMI, blood pressure,age, and insulin levels. Used classification algorithms on the Diabetes dataset to build and evaluate the model.


Description:
This project predicts whether a patient is **likely to have diabetes** based on health-related features from the **Pima Indians Diabetes dataset** using a Logistic Regression model.



## 📌 Project Overview
Diabetes is a chronic disease that can lead to serious health complications if not diagnosed early.  
Early prediction can help in prevention and treatment planning.  

In this project, we:
- Load and preprocess the dataset
- Train a *Logistic Regression* model
- Evaluate its performance using metrics like accuracy, precision, recall, and F1-score
- Create a prediction pipeline for new patient data

---

## 📂 Dataset
- **Source:** [Pima Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Instances:** 768 samples  
- **Features:** 8 health-related measurements (e.g., glucose level, BMI, age)  
- **Target:**  
  - 0 → No Diabetes  
  - 1 → Diabetes  

---

## 🛠️ Technologies Used
- *Python 3.x*
- *Pandas* – data manipulation  
- *NumPy* – numerical computations  
- *Matplotlib / Seaborn* – data visualization  
- *Scikit-learn* – model building and evaluation  

---

## 📊 Model Performance
| Metric      | Score |
|-------------|-------|
| Accuracy    | 77%   |
| Precision   | 74%   |
| Recall      | 69%   |
| F1-Score    | 71%   |

> The model provides decent accuracy and recall, making it useful as a preliminary screening tool.


## 📈 Visualization
- **Feature distribution** to understand data spread.  
- **Correlation heatmap** to identify relationships between features.  
- **Confusion matrix** to evaluate classification results.  

---

## 🔮 Future Improvements
- Deploy as a **Streamlit web app** for real-time predictions  
- Try advanced models like **Random Forest** or **XGBoost**  
 
