import gradio as gr
import numpy as np
import pandas as pd
import pickle

from train_model.train import *
# Load the pre-trained classifier model
with open("C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/models/cvd_model.pickle", "rb") as f:
    classifier_model = pickle.load(f)

# Function to make predictions using the classifier model
def predict_cardio_disease(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    # Preprocess input features if needed
    feature_names = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    input_data = {
        'age': age,
        'gender': 1 if gender == "Male" else 2,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': 0 if cholesterol == "Normal" else (1 if cholesterol == "Above Normal" else 3 ) ,
        'gluc': 0 if gluc == "Normal" else (1 if gluc == "Above Normal" else 3 ),
        
        'smoke': 0 if smoke == "Non Smoker" else 1,
        'alco':0 if alco == "No Alcohol" else 1,
        'active': 0 if active == "Non Active" else 1,
    }
    # Convert input data to a DataFrame with a single row
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]
    # print(input_df)
    
    # for col in feature_names:
    #     input_df[col] = pd.to_numeric(input_df[col])
        
    input_df = input_df.fillna(input_df.median())
    input_df = input_df.drop_duplicates()
    prediction = classifier_model.predict(input_df)
    
    # Return prediction result
    if prediction[0] == 0:
        result = "No Cardiovascular Disease Detected"
    else:
        result = "Cardiovascular Disease Detected"
        
    return result

# Create Gradio interface
def app_interface():
    with gr.Blocks() as interface:
        with gr.Row("Cardiovascular Disease Prediction"):
            gr.HTML("<img src='keensight_logo.png' alt='Company Logo'>")
            with gr.Column("Model Training 🧠"):
                gr.HTML("<h2>Train your own model!</h2>")
                parameters = [
                    gr.Slider(minimum=10, maximum=500, step = 5, label="Number of Estimators"),
                    gr.Slider(minimum=0.000000000000000000000000000000000000000000000000000000000001, maximum=1, label="Learning Rate"),
                    gr.Slider(minimum=0, maximum=1000, label="Max Depth"),
                    
                ]
                results = [
                    gr.Textbox(label="Accuracy Score"),
                    gr.Textbox(label="Precision Score"),
                    gr.Textbox(label="Recall Score"),
                    gr.Textbox(label="F1 Score"),
                ]
                train_button = gr.Button(value="Train Model")
            with gr.Column("Please fill the form to predict cardiovascular disease!"):
                gr.HTML("<h2>Please fill the form to predict cardiovascular disease!</h2>")
                inp = [
                    gr.Slider(label="Age", minimum=1, maximum=120),
                    gr.Radio(label="Gender", choices=["Male", "Female"]),
                    gr.Slider(label="Height (cm)", minimum=50, maximum=250, step=1),
                    gr.Slider(label="Weight (kg)", minimum=10, maximum=300, step=1),
                    gr.Slider(label="Systolic Blood Pressure", minimum=0, maximum=300, step=1),
                    gr.Slider(label="Diastolic Blood Pressure", minimum=0, maximum=200, step=1),
                    gr.Radio(label="Cholesterol Level", choices=["Normal", "Above Normal", "Well Above Normal"]),
                    gr.Radio(label="Glucose Level", choices=["Normal", "Above Normal", "Well Above Normal"]),
                    gr.Radio(label="Smoking", choices=["Non Smoker", "Smoker"]),
                    gr.Radio(label="Alcohol Consumption", choices=["No Alcohol", "Alcohol"]),
                    gr.Radio(label="Physical Activity", choices=["Non Active", "Active"]),
                ]
                
                output = [gr.Textbox(label="Prediction")]
                predict_button = gr.Button(value="Cardiovascular Disease Prediction")
        train_button.click(run, inputs=parameters, outputs=results)
        predict_button.click(predict_cardio_disease, inputs=inp, outputs=output)

    interface.launch()

if __name__ == "__main__":
    app_interface()
