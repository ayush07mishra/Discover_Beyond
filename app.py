from flask import Flask, request, jsonify
import joblib  # For scikit-learn models
# import tensorflow as tf  # For TensorFlow models
import logging

app = Flask(__name__)

# Enable logging for detailed errors
logging.basicConfig(level=logging.DEBUG)

# Load the machine learning model (update the path based on your model's location)
model = joblib.load('model.pkl')  # For scikit-learn models
# model = tf.keras.models.load_model('model.h5')  # For TensorFlow models

# Example function for recommendation (use your ML model here)
def recommend_country(best_time, people_travel_with, days_to_travel):
    try:
        # Map "best_time" from string (e.g., "Summer") to a number (e.g., 1 for Summer, 2 for Winter)
        time_mapping = {"Summer": 1, "Winter": 2}  # Example mapping
        best_time_numeric = time_mapping.get(best_time, 0)  # Default to 0 if no match

        # Convert the inputs to the format your model expects
        inputs = [[best_time_numeric, int(people_travel_with), int(days_to_travel)]]  # Example input format

        # Make prediction using your model
        prediction = model.predict(inputs)
        return prediction[0]  # Example: The model predicts the country name
    except Exception as e:
        app.logger.error(f"Error in recommend_country: {str(e)}")
        raise e

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        # Get data from the frontend (JSON)
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")

        best_time = data.get('best_time')
        people_travel_with = data.get('people_travel_with')
        days_to_travel = data.get('days_to_travel')

        # Ensure valid inputs (you can add more checks if needed)
        if not best_time or not people_travel_with or not days_to_travel:
            return jsonify({'error': 'Invalid input'}), 400

        # Get the recommendation using the ML model
        recommended_country = recommend_country(best_time, people_travel_with, days_to_travel)
    
        # Return the recommended country as a JSON response
        return jsonify({'recommended_country': recommended_country})

    except Exception as e:
        app.logger.error(f"Error in get_recommendation: {str(e)}")
        return jsonify({'error': f"Error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
