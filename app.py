import numpy as np
import gradio as gr
from catboost import CatBoostClassifier

clf = CatBoostClassifier()
clf.load_model('./model.bin')
               
def predict(pclass:int = 3,
    sex:str = 'male',
    age:float = 30,
    fare:float = 100,
    embarked:str = 'S'):

    prediction_array = np.array([pclass, sex, age, fare, embarked])
    survived = clf.predict(prediction_array)

    if survived == 1:
        return f"The Passenger Survived"
    else:
        return f"The Passenger did not Survive"
    

with gr.Blocks() as demo:
    #keeping the three categorical feature input in the same row
    with gr.Row() as row1:
        pclass = gr.Dropdown(choices=[1,2,3], label = 'passengerclass')
        sex = gr.Dropdown(choices=['male','female'], label = 'sex')
        embarked = gr.Dropdown(choices = ['C','Q','S'], label = 'embarked')

    # Creating slider for the two numerical inputs and also defining the limits for both

    age = gr.Slider(1,100, label = 'age', interactive = True)

    fare = gr.Slider(1,100, label = 'fare', interactive = True)

    submit = gr.Button(value = 'Pedict')

    # Showing the output
    output = gr.Textbox(label = 'whether the passenger survived?',interactive = False,)

    #Defining what happens when the user clicks the submit button
    submit.click(predict, inputs = [pclass, sex, age, fare, embarked], outputs = [output])

demo.launch(share = False, debug = False)
