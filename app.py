from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("best_model.pkl")

# Same encoding used during training
ENCODE = {
    'Gender':        {'Male': 1, 'Female': 0},
    'Married':       {'Yes': 1, 'No': 0},
    'Dependents':    {'0': 0, '1': 1, '2': 2, '3+': 3},
    'Education':     {'Graduate': 0, 'Not Graduate': 1},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
}

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.form

        gender        = ENCODE['Gender'][f['Gender']]
        married       = ENCODE['Married'][f['Married']]
        dependents    = ENCODE['Dependents'][f['Dependents']]
        education     = ENCODE['Education'][f['Education']]
        self_employed = ENCODE['Self_Employed'][f['Self_Employed']]
        applicant_inc = float(f['ApplicantIncome'])
        coapplicant   = float(f['CoapplicantIncome'])
        loan_amount   = float(f['LoanAmount'])
        loan_term     = float(f['Loan_Amount_Term'])
        credit_hist   = float(f['Credit_History'])
        property_area = ENCODE['Property_Area'][f['Property_Area']]

        features = np.array([[gender, married, dependents, education,
                               self_employed, applicant_inc, coapplicant,
                               loan_amount, loan_term, credit_hist,
                               property_area]])

        prediction = model.predict(features)[0]
        # LabelEncoder: N=0 → Rejected, Y=1 → Approved
        result = "✅ Loan Approved!" if prediction == 1 else "❌ Loan Rejected."

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)