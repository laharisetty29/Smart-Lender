# Smart Lender

## AI-Powered Loan Approval Prediction System

Smart Lender is a Machine Learning-based web application that helps banks and financial institutions make data-driven loan approval decisions. The system analyzes applicant information such as income, employment status, education, credit history, loan amount, and property area to predict whether a loan application is likely to be approved.

The application uses multiple machine learning algorithms and selects the best-performing model for real-time loan eligibility prediction through an interactive Flask web interface.

---

## Live Demo

🔗 https://smart-lender-1.onrender.com 

---

## GitHub Repository

🔗 https://github.com/laharisetty29/Smart-Lender

---

## Features

- Loan Approval Prediction
- User-Friendly Web Interface
- Real-Time Predictions
- Multiple ML Model Comparison
- Feature Importance Analysis
- Data Visualization Dashboard
- Flask-Based Deployment
- Pre-trained Model Integration

---

## Problem Statement

Banks receive thousands of loan applications every year. Manual verification is time-consuming and can lead to inconsistent decisions.

Smart Lender automates the loan eligibility assessment process by analyzing applicant details and predicting loan approval status using Machine Learning.

This helps:

- Reduce manual effort
- Improve decision-making speed
- Maintain consistency
- Support data-driven lending decisions

---

## Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Web Framework

- Flask

### Deployment

- Render

### Model Storage

- Pickle

---

## Project Structure

```text
Smart-Lender/
│
├── app.py
├── train_model.py
├── verify_model.py
├── best_model.pkl
├── requirements.txt
├── Procfile
├── .gitignore
│
├── templates/
│   └── index.html
│
├── confusion_matrix.png
├── credit_history.png
├── feature_importance.png
├── income_dist.png
├── loan_status_dist.png
└── model_comparison.png
```

---

## Dataset Information

The project uses the Loan Prediction Dataset containing applicant information and loan approval status.

### Features

| Feature | Description |
|----------|-------------|
| Gender | Male/Female |
| Married | Yes/No |
| Dependents | Number of Dependents |
| Education | Graduate/Not Graduate |
| Self_Employed | Yes/No |
| ApplicantIncome | Applicant Income |
| CoapplicantIncome | Co-applicant Income |
| LoanAmount | Requested Loan Amount |
| Loan_Amount_Term | Loan Repayment Term |
| Credit_History | Credit Score Availability |
| Property_Area | Urban/Semiurban/Rural |
| Loan_Status | Approved/Not Approved |

---

## Machine Learning Workflow

### Step 1: Data Collection

Load the loan prediction dataset.

### Step 2: Data Preprocessing

- Handle missing values
- Encode categorical features
- Normalize data if required

### Step 3: Exploratory Data Analysis

- Loan Status Distribution
- Income Distribution
- Credit History Analysis
- Feature Correlation Analysis

### Step 4: Model Training

Multiple machine learning models are trained and compared.

Examples:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression

### Step 5: Model Evaluation

Evaluation metrics:

- Accuracy Score
- Confusion Matrix
- Feature Importance
- Classification Performance

### Step 6: Model Selection

Best-performing model is saved as:

```python
best_model.pkl
```

### Step 7: Flask Integration

The trained model is integrated into a Flask application for real-time prediction.

### Step 8: Deployment

Application deployed using Render.

---

## Visualizations

The project includes several visual analysis reports:

### Loan Status Distribution

Shows approved and rejected loan applications.

### Income Distribution

Displays applicant income patterns.

### Credit History Analysis

Highlights the impact of credit history on loan approval.

### Feature Importance

Identifies the most influential features used by the model.

### Confusion Matrix

Measures prediction performance.

### Model Comparison

Compares the accuracy of different machine learning models.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/laharisetty29/Smart-Lender.git
```

### Move into Project Directory

```bash
cd Smart-Lender
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train_model.py
```

---

## Verify the Model

```bash
python verify_model.py
```

---

## Run the Flask Application

```bash
python app.py
```

---

## Sample Prediction Flow

### Input

- Gender: Male
- Married: Yes
- Education: Graduate
- Applicant Income: 5000
- Loan Amount: 150
- Credit History: 1
- Property Area: Urban

### Output

```text
Loan Approved
```

or

```text
Loan Not Approved
```

---

## Business Impact

### For Banks

- Faster loan processing
- Reduced operational workload
- Improved consistency in approvals

### For Customers

- Quick eligibility assessment
- Transparent decision support

### For Financial Institutions

- Better risk management
- Data-driven lending decisions

---

## Future Enhancements

- Streamlit Dashboard
- Explainable AI (XAI)
- Credit Score Integration
- Cloud Database Support
- Loan Risk Scoring
- AI-Powered Recommendation System
- Real-Time Banking API Integration

---

## Results

The trained model successfully predicts loan approval status based on applicant information and demonstrates strong classification performance for automated loan screening.

---

## Author

### Gadamsetty Lahari

B.Tech – Computer Science and Engineering (Data Science)

GitHub:
https://github.com/laharisetty29

LinkedIn:
https://www.linkedin.com/in/laharigadamsetty

---

## License

This project is developed for educational and learning purposes.

© 2026 Lahari Gadamsetty
