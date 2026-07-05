# 💳 Credit Card Fraud Detection App

A machine learning-powered web application that predicts the likelihood of credit card transactions being fraudulent using XGBoost classification.

## 🎯 Features

- **Real-time Fraud Prediction**: Instantly classify transactions as fraudulent or legitimate
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience
- **Visual Analytics**: Gauge chart displaying fraud probability with color-coded risk levels
- **Comprehensive Input**: Accepts transaction time, amount, and 28 PCA-transformed features (V1-V28)
- **High Performance**: Uses XGBoost model trained on a large-scale credit card dataset with SMOTE for handling class imbalance

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit** - Web application framework
- **XGBoost** - Machine learning model
- **scikit-learn** - Data preprocessing and scaling
- **Plotly** - Interactive visualizations
- **Joblib** - Model serialization
- **NumPy** - Numerical operations
- **imbalanced-learn (SMOTE)** - Handling class imbalance

## 📋 Project Structure

```
credit_card_det/
├── credit_card_pred.ipynb   # Jupyter notebook for model training and evaluation
├── credpred.py              # Streamlit web application
├── fraud_model.pkl          # Trained XGBoost model
├── scaler.pkl               # Trained StandardScaler for feature normalization
├── creditcard.csv           # Dataset (credit card transactions)
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup Steps

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd credit_card_det
   ```

2. **Install required dependencies**
   ```bash
   pip install streamlit xgboost scikit-learn pandas numpy matplotlib plotly joblib imbalanced-learn
   ```

3. **Verify model files exist**
   - Ensure `fraud_model.pkl` and `scaler.pkl` are present in the project directory

## 💻 Usage

### Running the Web Application

```bash
streamlit run credpred.py
```

The application will open in your default web browser at `http://localhost:8501`.

### How to Use

1. **Enter Transaction Details**:
   - **Time**: Transaction time in seconds from the reference time
   - **Amount**: Transaction amount in euros (€)
   - **PCA Features (V1-V28)**: Click "Enter PCA Feature Values (V1-V28)" to expand and input the 28 anonymized features

2. **Click "🔍 Predict Fraud"** to get the prediction results

3. **View Results**:
   - Fraud detection status (🚨 Fraud Detected or ✅ Legitimate Transaction)
   - Confidence percentage
   - Interactive gauge chart showing fraud probability

## 🧠 Model Details

### Training Process

The model was trained in `credit_card_pred.ipynb` with the following workflow:

1. **Data Loading**: Used the `creditcard.csv` dataset containing anonymized credit card transactions
2. **Preprocessing**: 
   - Separated features (Time, Amount, V1-V28) and target variable (Class)
   - Applied `StandardScaler` for feature normalization
3. **Handling Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) with sampling strategy of 0.2 to address the imbalanced dataset
4. **Train-Test Split**: 80% training, 20% testing with stratification
5. **Model Training**: XGBoost Classifier with the following hyperparameters:
   - `n_estimators`: 300
   - `learning_rate`: 0.05
   - `max_depth`: 6
   - `subsample`: 0.8
   - `colsample_bytree`: 0.8
6. **Evaluation Metrics**:
   - Classification Report
   - Confusion Matrix
   - ROC-AUC Score
   - Precision-Recall AUC

### Model Performance

The trained model achieves strong performance metrics including:
- High precision and recall for fraud detection
- Robust ROC-AUC and Precision-Recall AUC scores
- Effective handling of class imbalance through SMOTE

## 📊 Dataset Information

The project uses a credit card transaction dataset with the following features:
- `Time`: Number of seconds elapsed from the first transaction
- `Amount`: Transaction amount
- `V1-V28`: PCA-transformed features (anonymized for privacy)
- `Class`: Target variable (0 = legitimate, 1 = fraudulent)

## 🎨 UI Features

- **Responsive Design**: Wide layout optimized for desktop screens
- **Custom Styling**: Professional appearance with custom colors and card components
- **Interactive Elements**: Expandable sections for advanced inputs
- **Color-Coded Results**:
  - 🔴 Red: High fraud probability
  - 🟡 Yellow: Medium risk
  - 🟢 Green: Low risk/legitimate

## 🔧 Development

### Retraining the Model

To retrain the model with updated data:

1. Open `credit_card_pred.ipynb`
2. Update the dataset path if needed
3. Adjust hyperparameters or SMOTE sampling strategy as required
4. Run all cells to train and save the new model:
   ```python
   joblib.dump(xgb, "fraud_model.pkl")
   joblib.dump(scaler, "scaler.pkl")
   ```
5. The Streamlit app will automatically use the updated model files

## 📦 Dependencies

See `credit_card_pred.ipynb` for specific package versions. Core dependencies include:

- streamlit
- xgboost
- scikit-learn
- pandas
- numpy
- matplotlib
- plotly
- joblib
- imbalanced-learn

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. While the model is trained on real credit card transaction data, it should **not** be used for production financial decision-making without proper validation, security measures, and compliance with financial regulations.

## 👨‍💻 Author

Built with Streamlit and XGBoost

## 📄 License

This project is open source and available for educational use.