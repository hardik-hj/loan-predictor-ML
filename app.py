import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, session

# Create Flask app
app = Flask(__name__)

# Secret key for session handling (make sure to change this in production)
app.secret_key = 'your_secret_key_here'

# Load your model and PCA object
model = pickle.load(open('model.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb')) 

# Route the home page
@app.route('/')
def home():
    # Set dark mode state if it's in the session, otherwise default to False
    dark_mode = session.get('dark_mode', False)
    return render_template('index.html', dark_mode=dark_mode)

# Route the predict URL
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Extract features from form input
        iFeatures = [float(data) for data in request.form.values()]
        iFeatures = np.array(iFeatures).reshape(1, -1)  # Ensure input is a 2D array
        
        # Transform input features using the pre-fitted PCA
        input_transformed = pca.transform(iFeatures)
        
        # Predict using the loaded model
        prediction = model.predict(input_transformed)

        if prediction[0] == 0.0:
            prediction_text = "The person is most likely to pay off the loan."
        elif prediction[0] == 1.0:
            prediction_text = "The person is not likely to repay the loan."
        else:
            prediction_text = "The prediction result is unclear."

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            prediction_class='success' if prediction[0] == 0 else 'error',
            **request.form,  # Pass form data back to the template
            dark_mode=session.get('dark_mode', False)  # Pass dark mode state
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"An error occurred: {e}",
            **request.form,  # Pass form data back to the template even if an error occurs
            dark_mode=session.get('dark_mode', False)  # Pass dark mode state
        )

# Route to toggle dark mode
@app.route('/toggle_dark_mode', methods=['POST'])
def toggle_dark_mode():
    dark_mode = not session.get('dark_mode', False)  # Toggle dark mode state
    session['dark_mode'] = dark_mode  # Store dark mode state in session
    return jsonify({'dark_mode': dark_mode})

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # This allows external access
  # Disable debug mode in production
