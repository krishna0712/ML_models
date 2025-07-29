# 🚢 Titanic Survival Prediction using Random Forest

This project predicts whether a passenger survived the Titanic disaster based on attributes like age, sex, class, etc., using a machine learning classification approach.

---

## 📂 Dataset

- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Features include:
  - `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, etc.
  - Target variable: `Survived` (0 = No, 1 = Yes)

---

## 🧪 Final Model Performance

| Metric       | Train Set | Test Set |
|--------------|-----------|----------|
| Accuracy     | 98%       | 99%      |

### 🔢 Confusion Matrix

  [[119 0]
  [ 2 58]]


### 📊 Classification Report


          precision    recall  f1-score   support

       0       0.98      1.00      0.99       119
       1       1.00      0.97      0.98        60

accuracy                           0.99       179


---

## 🔧 Project Steps

### 1. 📥 Importing and Loading Data
- Loaded data using `pandas`
- Imported all necessary ML libraries (`sklearn`, `numpy`, etc.)

### 2. 🧹 Data Cleaning
- Dropped columns not useful for prediction:
  - `Name`, `Cabin`, `Ticket`, `PassengerId`
- Visualized null values and filled them:
  - `Age`: filled using **mean**
  - `Embarked`: filled using **mode**

### 3. 🏷️ Encoding Categorical Data
- Applied `LabelEncoder` to:
  - `Sex` → 0 for female, 1 for male
  - `Embarked` → converted C/Q/S to 0/1/2

✅ Now all features were converted to **numerical** format.

### 4. 📊 Exploratory Analysis (Optional)
- Printed `.value_counts()` and `.describe()` for key features
- Checked correlation (optional)

### 5. 🧪 Train-Test Split
```python```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

### 6. Model Training and Evaluation

-Trained a Random Forest Classifier, known for its robustness and ability to handle feature interactions.
-Evaluated the model using accuracy score, confusion matrix, and classification report.


### 7️⃣ Final Results and Conclusion
  The model showed excellent generalization and minimal overfitting.
  All features were engineered and preprocessed effectively.
  Demonstrated the power of simple models when paired with thoughtful data cleaning.


### ✅ Manual Prediction Check:
Sample predictions on the test set matched expected outcomes, confirming that the model behaves correctly beyond just scoring metrics.
