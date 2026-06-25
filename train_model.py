import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\lahar\OneDrive\Documents\train_dataset.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ─── 2. EDA PLOTS ───────────────────────────────────────────────────────────
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status Distribution")
plt.tight_layout()
plt.savefig("loan_status_dist.png")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df['ApplicantIncome'], bins=40, kde=True)
plt.title("Applicant Income Distribution")
plt.tight_layout()
plt.savefig("income_dist.png")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Credit History vs Loan Status")
plt.tight_layout()
plt.savefig("credit_history.png")
plt.show()

# ─── 3. PREPROCESSING ───────────────────────────────────────────────────────
df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

print("\nAfter cleaning missing values:\n", df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nData after encoding:\n", df.head())

# ─── 4. SPLIT ────────────────────────────────────────────────────────────────
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ─── 5. TRAIN ALL MODELS ────────────────────────────────────────────────────
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5),
    "XGBoost":       XGBClassifier(use_label_encoder=False,
                                   eval_metric='logloss', random_state=42)
}

results = {}
best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    results[name] = {"Train Acc": round(train_acc*100,2),
                     "Test Acc":  round(test_acc*100,2)}
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test  Accuracy: {test_acc*100:.2f}%")
    print(classification_report(y_test, model.predict(X_test)))

    if test_acc > best_acc:
        best_acc   = test_acc
        best_model = model
        best_name  = name

# ─── 6. CONFUSION MATRIX FOR BEST MODEL ─────────────────────────────────────
print(f"\nBest Model: {best_name} ({best_acc*100:.2f}% test accuracy)")
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected','Approved'],
            yticklabels=['Rejected','Approved'])
plt.title(f"Confusion Matrix – {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ─── 7. MODEL COMPARISON PLOT ────────────────────────────────────────────────
names  = list(results.keys())
t_acc  = [results[n]["Train Acc"] for n in names]
te_acc = [results[n]["Test Acc"]  for n in names]

x = np.arange(len(names))
plt.figure(figsize=(8,5))
plt.bar(x-0.2, t_acc,  0.4, label='Train Accuracy', color='steelblue')
plt.bar(x+0.2, te_acc, 0.4, label='Test Accuracy',  color='coral')
plt.xticks(x, names)
plt.ylim(60, 100)
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ─── 8. SAVE BEST MODEL ──────────────────────────────────────────────────────
joblib.dump(best_model, "best_model.pkl")
print(f"\nSaved '{best_name}' as best_model.pkl")

# ─── 9. FEATURE IMPORTANCE (XGBoost / Random Forest) ────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns)
    fi.sort_values().plot(kind='barh', figsize=(7,5), color='teal')
    plt.title(f"Feature Importances – {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()