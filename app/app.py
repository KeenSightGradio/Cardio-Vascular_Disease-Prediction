# import gradio as gr
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pickle

# with open("C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/models/cvd_model.pickle", "rb") as f:
#     classifier_model = pickle.load(f)

# # Function to make predictions using the classifier model
# def predict_cardio_disease(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
#     # Preprocess input features if needed
#     feature_names = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
#     input_data = {
#     'age': age,
#     'gender': gender,
#     'height': height,
#     'weight': weight,
#     'ap_hi': ap_hi,
#     'ap_lo': ap_lo,
#     'cholesterol': cholesterol,
#     'gluc': gluc,
#     'smoke': smoke,
#     'alco': alco,
#     'active': active
# }
#     # Convert input data to a DataFrame with a single row
#     input_df = pd.DataFrame([input_data])
#     input_df = input_df[feature_names]
#     for col in feature_names:
#         input_df[col] = pd.to_numeric(input_df[col])
        
#     input_df = input_df.fillna(input_df.median())
#     input_df=input_df.drop_duplicates()
#     prediction = classifier_model.predict(input_df)
    
#     # Return prediction result
#     if prediction[0] == 0:
#         result = "No Cardiovascular Disease Detected"
#     else:
#         result = "Cardiovascular Disease Detected"
        
#     return result

# # Input components
# age_input = gr.Number(label="Age", minimum=1, maximum=120)
# gender_input = gr.Dropdown(label="Gender", choices=[("Male", 1), ("Female", 2)])
# height_input = gr.Slider(label="Height (cm)", minimum=50, maximum=250, step=1)
# weight_input = gr.Slider(label="Weight (kg)", minimum=10, maximum=300, step=1)
# ap_hi_input = gr.Slider(label="Systolic Blood Pressure", minimum=0, maximum=300, step=1)
# ap_lo_input = gr.Slider(label="Diastolic Blood Pressure", minimum=0, maximum=200, step=1)
# cholesterol_input = gr.Dropdown(label="Cholesterol Level", choices=[("Normal", 1), ("Above Normal",2), ("Well Above Normal", 3)])
# gluc_input = gr.Dropdown(label="Glucose Level",choices=[("Normal", 1), ("Above Normal",2), ("Well Above Normal", 3)])
# smoke_input = gr.Dropdown(label="Smoking", choices=[("Non Smoker", 0), ("Smoker", 1)])
# alco_input = gr.Dropdown(label="Alcohol Consumption", choices=[("No Alcohol",0), ("Alcohol",1)])
# active_input = gr.Dropdown(label="Physical Activity", choices=[("Non Active", 0), ("Active", 1)])

# # Output component
# result_output = gr.Textbox(label="Prediction")


# # Input components
# column1 = gr.Group([age_input, gender_input, height_input, weight_input])
# column2 = gr.Group([ap_hi_input, ap_lo_input, cholesterol_input, gluc_input])
# column3 = gr.Group([smoke_input, alco_input, active_input])


# # Create Gradio interface
# gr.Interface(fn=predict_cardio_disease, 
#              inputs=[age_input, gender_input, height_input, weight_input, ap_hi_input, ap_lo_input, 
#                      cholesterol_input, gluc_input, smoke_input, alco_input, active_input], 
#              outputs=result_output,
#              title="Cardiovascular Disease Prediction",
#              description="Enter patient details to predict cardiovascular disease.",
#             # layout="horizontal",
#             css=".input-column { display: inline-block; width: 200px; margin-right: 20px; }").launch()


