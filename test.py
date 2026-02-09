from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data
    input_data = [
        float(request.form['age_first_funding_year']),
        float(request.form['age_last_funding_year']),
        float(request.form['age_first_milestone_year']),
        float(request.form['age_last_milestone_year']),
        float(request.form['relationships']),
        float(request.form['funding_rounds']),
        float(request.form['funding_total_usd']),
        float(request.form['milestones']),
        float(request.form['avg_participants'])
    ]

    # Prediction
    prediction = model.predict([input_data])[0]

    # Result mapping
    result = "Acquired" if prediction == 1 else "Closed"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
