import gradio as gr
import numpy as np
import pandas as pd
import pickle

from train_model.train import *
# Load the pre-trained classifier model
with open("C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/models/cvd_model.pickle", "rb") as f:
    classifier_model = pickle.load(f)
def load_dataset_for_multipredcition(dataset_path):
    not_clean = pd.read_csv(dataset_path) 
    heart_disease_data = not_clean.iloc[:,0].apply(lambda x: pd.Series(str(x).split(";")))
    # print(heart_disease_data)
    heart_disease_data.columns = ["id","age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active",]

    column_names = ["id","age","gender","height","weight",
                "ap_hi","ap_lo","cholesterol","gluc",
                "smoke","alco","active"]
    
    for col in column_names:
        heart_disease_data[col] = pd.to_numeric(heart_disease_data[col])
    # save the ids
    ID = heart_disease_data['id']
    # Replace missing values with column median
    heart_disease_data = heart_disease_data.fillna(heart_disease_data.median())
    heart_disease_data=heart_disease_data.drop(labels="id", axis=1)
    heart_disease_data=heart_disease_data.drop_duplicates()

    return ID, heart_disease_data
    
def multi_predict(input_df):
    input_df = input_df.fillna(input_df.median())
    input_df = input_df.drop_duplicates()
    prediction = classifier_model.predict(input_df)
    
    # Return prediction result
    if prediction[0] == 0:
        result = "No Cardiovascular Disease Detected ✅"
    else:
        result = "Cardiovascular Disease Detected ⚠️"
    return result


def multi_prediction(dataset_path):
    Id_tags, heart_disease_data = load_dataset_for_multipredcition(dataset_path)
    feature_names = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    
    # print(type(Id_tags))
    predictions = []
    
    for index, row in heart_disease_data.iterrows():
        input_data = {
        'age': row['age'] // 365,
        'gender': row['gender'],
        'height': row['height'],
        'weight': row['weight'],
        'ap_hi': row['ap_hi'],
        'ap_lo': row['ap_lo'],
        'cholesterol': row['cholesterol'],
        'gluc': row['gluc'],
        'smoke': row['smoke'],
        'alco':row['alco'],
        'active': row['active']
    }
    
    # Convert input data to a DataFrame with a single row
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        
        predict = multi_predict(input_df)
        predictions.append(predict)
        
    output_tuples = list(zip(Id_tags, predictions))
    # Save the output to a text file
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write("ID, Prediction\n")
        for id_value, prediction in output_tuples:
            file.write(f"{id_value}, {prediction}\n")

    # Read the contents of the file
    with open("output.txt", "r", encoding="utf-8") as file:
        output_text = file.read()

    return output_text
    # return predictions
    
# Function to make predictions using the classifier model
def predict_cardio_disease(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    # Preprocess input features if needed
    feature_names = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    input_data = {
        'age': age // 365,
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
        result = "No Cardiovascular Disease Detected ✅"
    else:
        result = "Cardiovascular Disease Detected ⚠️"
        
    return result

# Create Gradio interface
def app_interface():
    with gr.Blocks() as interface:
        
        # gr.HTML("<img src='https://i.ibb.co/Bw08434/logo-1.png' alt='Logo' style='width:230px;height:100px;border-radius:5px;box-shadow:2px 2px 5px 0px rgba(0,0,0,0.75);background-color:black;'><br>",)
    
        # with gr.Row("Cardiovascular Disease Prediction"):
        #     gr.Image("logo_keensight.png", height=100, width=300)
        with gr.Row(""):  
            
            with gr.Column("Model Training"):
                gr.Image("logo_keensight.png", height=100, width=300)
                gr.HTML("<h2>Train your own model ❤️!</h2>")
                parameters = [
                    gr.Slider(minimum=5, maximum=500, step = 5, label="Number of Estimators"),
                    gr.Slider(minimum=0.00000000001, maximum=1, label="Gamma", step = 0.2),
                    gr.Slider(minimum=0.000000000001, maximum=1, label="Learning Rate"),
                    gr.Slider(minimum=5, maximum=100, label="Max Depth", step = 1),
                    gr.Slider(minimum=0.00000000001, maximum=1, label="Test Size", step= 0.1)
                    
                ]
                results = [
                    gr.Textbox(label="Accuracy Score"),
                    gr.Textbox(label="Precision Score"),
                    gr.Textbox(label="Recall Score"),
                    gr.Textbox(label="F1 Score"),
                    gr.Gallery(allow_preview=True, label="Data Visualization", object_fit="fill", type="numpy", height="auto", rows=(1, 2), columns=[1])
                    
                    # gr.Image(label="ROC Curve"),
                    # gr.Image(label="Learning Curve")

                ]
                train_button = gr.Button(value="Train Model")
                gr.HTML("<h3>Dataset link here: <a href='https://www.kaggle.com/datasets/bhadaneeraj/cardio-vascular-disease-detection'>Dataset</a>.</h3>")
            
                
            with gr.Column("Please fill the form to predict cardiovascular disease!"):
                gr.HTML("<h2>Please fill the form to predict cardiovascular disease ❤️!</h2>")
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
            
            with gr.Column(""):
                gr.HTML("<h2>Upload CSV file to predict!</h2><h4>Please make sure your CSV file has the following columns in the correct naming and order:</h4> <br> <ol> <li>id - an integer</li><li>age - in number of days</li> <li>gender - 1 for male, 0 otherwise</li>  <li>height - in centimeters</li>  <li>weight - in Kg</li>  <li>ap_hi - Systolic blood pressure - int </li>  <li>ap_lo - Diastolic blood pressure - int</li>  <li>cholesterol - 1: normal, 2: above normal, 3: well above normal </li> <li>gluc -  1: normal, 2: above normal, 3: well above normal</li>   <li>smoke - binary - 0 or 1</li>   <li>alco - binary - 0 or 1</li>   <li>active - binary - 0 or 1</li> </ol>")
                parameter = [
                     gr.File(file_types=[".csv"])
                ]
                res = [
                    gr.Text(label="Predictions", type="text")
                ]
                pred_button = gr.Button(value="Predict")
                
                
        train_button.click(run, inputs=parameters, outputs=results)
        pred_button.click(multi_prediction, inputs=parameter, outputs=res)
        predict_button.click(predict_cardio_disease, inputs=inp, outputs=output)

    interface.launch()

if __name__ == "__main__":
    app_interface()
