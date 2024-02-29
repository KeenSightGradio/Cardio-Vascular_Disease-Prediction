# Cardiovascular Disease Prediction

This is a Gradio application that predicts the presence or absence of cardiovascular disease based on patient details. It utilizes a pre-trained Random Forest Classifier model.

## Installation

To run the application, you need to have the following dependencies installed:

- Python 3.7 or higher
- pandas
- scikit-learn
- gradio

You can install the dependencies using the following command:

## Usage

1. Clone the repository or download the code files.

2. Open a terminal or command prompt and navigate to the project directory.

3. Run the following command to start the application:

4. The application will start and display a web address (e.g., http://127.0.0.1:7860). Open the address in a web browser.

5. Enter the patient details in the input fields provided, such as age, gender, height, weight, blood pressure, cholesterol level, glucose level, smoking status, alcohol consumption, and physical activity.

6. Click the "Predict" button to generate the prediction.

7. The prediction result, indicating whether cardiovascular disease is detected or not, will be displayed in the output box.

## Model and Preprocessing

The application uses a pre-trained Random Forest Classifier model for prediction. The model file, `cvd_model.pickle`, is located in the `models` directory.

The input data is preprocessed before making predictions. The preprocessing includes handling missing values, converting categorical features to numeric, and scaling numeric features.

## Customization

If you want to customize or modify the application, you can make changes to the following files:

- `app.py`: Contains the main code for the Gradio application, including the prediction function and the user interface layout.

- `cvd_model.pickle`: The pre-trained Random Forest Classifier model file. You can replace this file with your own trained model if desired.

Feel free to explore and modify the code to suit your specific requirements.

