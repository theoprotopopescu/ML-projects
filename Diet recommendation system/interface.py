import gradio as gr
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

def recommend_top5_meals_by_type(patient_data, encoders, scaler, model, meal_columns, numerical_features, patient_features):
    """
    Generate the top 5 meal recommendations for each meal type (e.g., breakfast, lunch, dinner).

    Args:
        patient_data (dict): A dictionary containing the patient's information. Keys should match the column names.
        encoders (dict): A dictionary of LabelEncoders for meal columns (e.g., breakfast, lunch, dinner).
        scaler (StandardScaler): A fitted StandardScaler for normalizing numerical features.
        model (tf.keras.Model): The trained recommendation model.
        meal_columns (list): List of meal columns (e.g., ['BREAKFAST', 'LUNCH', 'DINNER']).
        numerical_features (list): List of numerical feature column names.
        patient_features (list): List of all feature column names required by the model.

    Returns:
        dict: A dictionary with meal types as keys and a list of top 5 recommended meals as values.
    """
    # Step 1: Encode and normalize the patient data
    df = pd.DataFrame([patient_data])  # Convert patient_data into a DataFrame

    df[numerical_features] = scaler.transform(df[numerical_features])  # Normalize numerical features

    # Step 2: Prepare patient input as a NumPy array
    patient_input = np.array(df[patient_features])

    recommendations = {}

    # Step 3: Generate predictions for each meal type
    for meal_type in meal_columns:
        # Meal IDs for the current meal type
        meal_input = np.arange(len(encoders[meal_type].classes_)).reshape(-1, 1)  # Meal IDs from 0 to the number of meal classes
        patient_data = np.tile(patient_input, (len(meal_input), 1))
        
        # Predict probabilities for all meal options
        predictions = model.predict([patient_data, meal_input], verbose=0).flatten()

        # Identify the top 5 recommendations (highest probabilities)
        top5_meal_ids = predictions.argsort()[::-1][:5]  # Sort probabilities and get top 5 meal IDs
        top5_meals = encoders[meal_type].inverse_transform(top5_meal_ids)  # Decode meal IDs to original meal names

        recommendations[meal_type] = top5_meals

    return recommendations

# Prediction function
def recommend_meal(age, gender, bmi, height, weight, systolic_bp, diastolic_bp, glucose,
                    hba1c, cholesterol, triglycerides, egfr, smoker, ckd, ihd, hypertension, diabetes):
    """
    Predicts 5 meal plans based on health parameters using a machine learning model.
    (For now, it returns fixed sample meal plans.)
    """
    # Example call to the function
    patient_data = {
    'AGE': age,
    'GENDER': 1 if gender=="Male" else 0,
    'BMI': bmi,
    'HEIGHT': height,
    'WEIGHT': weight,
    'SYSTOLIC BLOOD PRESSURE': systolic_bp,
    'DIASTOLIC BLOOD PRESSURE': diastolic_bp,
    'GLUCOSE': glucose,
    'GLYCOSYLATED HEMOGLOBIN': hba1c,
    'TOTAL CHOLESTEROL': cholesterol,
    'TRIGLYCERIDES': triglycerides,
    'eGFR': egfr,
    'SMOKER': 1 if smoker=="Yes" else 0,
    'CHRONIC KIDNEY DISEASE': 1 if ckd else 0,
    'ISCHEMIC HEART DISEASE': 1 if ihd else 0,
    'HYPERTENSION': 1 if hypertension else 0,
    'DIABETES': 1 if diabetes else 0,
    }

    model = load_model('best_model.keras')
    encoders = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    # Get top 5 recommended meals for each meal type
    recommendations = recommend_top5_meals_by_type(
        patient_data=patient_data,
        encoders=encoders,
        scaler=scaler,
        model=model,
        meal_columns=['BREAKFAST', 'LUNCH', 'DINNER'],
        numerical_features=['AGE', 'BMI', 'HEIGHT', 'WEIGHT', 'SYSTOLIC BLOOD PRESSURE',
                            'DIASTOLIC BLOOD PRESSURE', 'GLUCOSE', 'GLYCOSYLATED HEMOGLOBIN',
                            'TOTAL CHOLESTEROL', 'TRIGLYCERIDES', 'eGFR'],
        patient_features=['AGE', 'GENDER', 'BMI', 'HEIGHT', 'WEIGHT', 'SYSTOLIC BLOOD PRESSURE',
                          'DIASTOLIC BLOOD PRESSURE', 'GLUCOSE', 'GLYCOSYLATED HEMOGLOBIN',
                          'TOTAL CHOLESTEROL', 'TRIGLYCERIDES', 'eGFR', 'SMOKER',
                          'CHRONIC KIDNEY DISEASE', 'ISCHEMIC HEART DISEASE', 'HYPERTENSION', 'DIABETES']
    )

    # Format recommendations as a string
    formatted_recommendations = "\n".join(
        f"{meal_type}:\n" + "\n".join(f"  Option {i + 1}: {meal}" for i, meal in enumerate(meals))
        for meal_type, meals in recommendations.items()
    )

    return formatted_recommendations


# Function to compute BMI
def compute_bmi(height, weight):
    if height > 0:  # Avoid division by zero
        bmi = weight / (height ** 2)
        return round(bmi, 2)
    return 0.0

# Gradio inputs
with gr.Blocks() as interface:
    gr.Markdown("# Diet Recommendation System")
    gr.Markdown("Enter your health parameters to get a personalized diet recommendation.")


    with gr.Column():
        gr.Markdown("### Personal Information")
        with gr.Row():
            age = gr.Number(label="Age", value=30)
            gender = gr.Dropdown(choices=["Male", "Female"], label="Gender")
        with gr.Row():
            height = gr.Number(label="Height (m)", value=1.70)
            weight = gr.Number(label="Weight (kg)", value=70)
        with gr.Row():
            bmi = gr.Number(label="BMI (Computed Automatically)", value=0.0, interactive=False)

    with gr.Column():
        gr.Markdown("### Health Metrics")
        with gr.Row():
            systolic_bp = gr.Number(label="Systolic Blood Pressure", value=120)
            diastolic_bp = gr.Number(label="Diastolic Blood Pressure", value=80)
        with gr.Row():
            glucose = gr.Number(label="Glucose (mg/dL)", value=90)
            hba1c = gr.Number(label="Glycosylated Hemoglobin (HbA1c %)", value=5.5)
        with gr.Row():
            cholesterol = gr.Number(label="Total Cholesterol (mg/dL)", value=180)
            triglycerides = gr.Number(label="Triglycerides (mg/dL)", value=150)
            egfr = gr.Number(label="eGFR (mL/min/1.73m²)", value=90)

    with gr.Column():
        gr.Markdown("### Health Conditions")
        with gr.Row():
            ckd = gr.Checkbox(label="Chronic Kidney Disease")
            ihd = gr.Checkbox(label="Ischemic Heart Disease")
        with gr.Row():
            hypertension = gr.Checkbox(label="Hypertension")
            diabetes = gr.Checkbox(label="Diabetes")
        with gr.Row():
            smoker = gr.Dropdown(choices=["Yes", "No"], label="Smoker")
        
    # Output at the bottom
    gr.Markdown("### Recommendations")
    output = gr.Textbox(label="Recommended Meal Plans", lines=10)

    # Button to submit
    submit_button = gr.Button("Get Recommendations")

    def update_bmi(height, weight):
        return gr.update(value=compute_bmi(height, weight))

    # Link BMI calculation to Height and Weight changes
    height.change(fn=update_bmi, inputs=[height, weight], outputs=bmi)
    weight.change(fn=update_bmi, inputs=[height, weight], outputs=bmi)

    # Link function to inputs
    submit_button.click(
        recommend_meal,
        inputs=[age, gender, bmi, height, weight, systolic_bp, diastolic_bp, glucose,
                hba1c, cholesterol, triglycerides, egfr, smoker, ckd, ihd, hypertension, diabetes],
        outputs=output
    )

# Launch the app
interface.launch()