# Yarn-Quality-Prediction-Model
Machine Learning Model for Yarn Quality Prediction This project uses Random Forest and Gradient Boosting Regressors to predict yarn properties (strength, elongation, CV, thin/thick places, neps, and hairiness) based on cotton blend data. It includes data preprocessing, scaling, model training, and ensemble averaging for improved accuracy.
## 📘 Overview
This project predicts **yarn quality properties** such as strength, elongation, CV, thin/thick places, neps, and hairiness using **machine learning regression models**.  
It applies an **ensemble approach** combining **Random Forest** and **Gradient Boosting** to achieve stable and accurate results.

The goal is to help textile manufacturers and researchers analyze and optimize **cotton blend properties** to improve yarn performance.

---

## 🧩 Dataset
- **File used:** `yarn_training_data_10000.csv`  
- **Source:** Cotton blend dataset (collected for yarn property analysis)  
- **Size:** 10,000 rows × 25 columns (approx.)
- **Key Features:**  
  - Cotton composition and fiber attributes (e.g., length, tenacity, short fiber %, etc.)  
- **Target Columns:**  
  ```
  yarn_strength  
  yarn_elongation  
  yarn_cv  
  yarn_thin_places  
  yarn_thick_places  
  yarn_neps  
  yarn_hairiness
  ```

---

## ⚙️ Model Details
This project uses **two ensemble models**:
1. **Random Forest Regressor** – robust against noise, handles feature interactions well.  
2. **Gradient Boosting Regressor** – captures complex relationships and reduces bias.  

### Training Process:
- Data split into **80% training** and **20% testing**.  
- **StandardScaler** used for feature scaling.  
- Each target property is trained **individually**.  
- Final prediction = **average of RF + GB outputs**.

---

## 📊 Results Summary
Model performance (example output printed in console):

| Target Property     | R² Score | Mean Absolute Error |
|----------------------|----------|---------------------|
| yarn_strength        | ~0.81    | Low error           |
| yarn_elongation      | ~0.12    | Higher error        |
| yarn_cv              | ~0.60    | Good fit            |
| yarn_thin_places     | ~0.42    | Moderate fit        |
| yarn_thick_places    | ~0.47    | Moderate fit        |
| yarn_neps            | ~0.35    | Acceptable          |
| yarn_hairiness       | ~0.58    | Good consistency    |

> Results vary depending on data randomness and model tuning.

---

## 🧠 Example Prediction
The model includes an example prediction for **one random blend sample** at the end of the script:
```python
sample_preds = {
  'yarn_strength': 18.22,
  'yarn_elongation': 4.11,
  'yarn_cv': 12.05,
  'yarn_thin_places': 11.4,
  'yarn_thick_places': 8.3,
  'yarn_neps': 9.6,
  'yarn_hairiness': 4.9
}
```

---

## 🖥️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/Yarn-Quality-Prediction.git
   cd Yarn-Quality-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Place your dataset file (`yarn_training_data_10000.csv`) in the same directory.  
4. Run the script:
   ```bash
   python csv_ml.py
   ```

---

## 🧩 Technologies Used
- **Python 3.10+**
- **Pandas**
- **NumPy**
- **Scikit-learn**

---

## ✨ Future Improvements
- Add **deep learning models** for comparison.  
- Include **feature importance visualization**.  
- Create a **GUI or web dashboard** for easier use.  

---

## 👤 Author
**Bavly George**  
- 🎓 Faculty of Computers and Data Science, Alexandria University  
- 💻 Skilled in Java, Python, R, and Machine Learning  
- 📫 [LinkedIn](#) | [GitHub](#)

---

## 📜 License
This project is open-source and available for educational and research purposes.
