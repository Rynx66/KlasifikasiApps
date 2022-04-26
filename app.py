import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash import no_update
import base64
import io
from io import BytesIO
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization, Activation
from keras import backend as K
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.models import load_model

classes = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

enc = OneHotEncoder()
enc.fit([[0], [1], [2], [3]]) 
def names(number):
    if(number == 0):
        return 'Glioma tumor'
    elif(number == 1):
        return 'Normal'
    elif(number == 2):
        return 'Meningioma tumor'
    elif(number == 3):
        return 'Pituitary tumor'  

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.title = 'Klasifikasi Apps'

app.layout = html.Div([


    html.H1(children='Klasifikasi Tumor Otak', style={'textAlign': 'center'
        }),
    html.H5(children='Create by Muhammad Fachry Nurmansyach', style={'textAlign': 'center'
        }),
    
    
    
    dcc.Markdown('''
                ###### Langkah Pertama  : Masukkan gambar !!
                ###### Langkah Kedua    : Tunggu proses Prediksi dan Output akan muncul !!
    '''),
    
    
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '95%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }, 
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload', style={'position':'absolute', 'left':'200px', 'top':'250px'}),
    
    html.Div(id='prediction', style={'position':'absolute', 'left':'800px', 'top':'310px', 'font-size':'x-large'}),
    html.Div(id='prediction2', style={'position':'absolute', 'left':'800px', 'top':'365px', 'font-size': 'x-large'}),
    
    html.Div(id='facts', style={'position':'absolute', 'left':'800px', 'top':'465px', 'font-size': 'large',\
                               'height': '200px', 'width': '500px'}),

])

def parse_contents(contents):
    
    return html.Img(src=contents, style={'height':'450px', 'width':'450px'})


@app.callback([Output('output-image-upload', 'children'), Output('prediction', 'children'), Output('prediction2', 'children'), 
              Output('facts', 'children')],
              [Input('upload-image', 'contents')])

def update_output(list_of_contents):        
    
    if list_of_contents is not None:
        children = parse_contents(list_of_contents[0]) 
         
        img_data = list_of_contents[0]
        img_data = re.sub('data:image/jpeg;base64,', '', img_data)
        img_data = base64.b64decode(img_data)  
        
        stream = io.BytesIO(img_data)
        img_pil = Image.open(stream)
        
        
        #Load model, change image to array and predict
        model = load_model('Hasil_Training.h5') 
        dim = (150, 150)
        
        img = np.array(img_pil.resize(dim))
        
        x = img.reshape(1,150,150,3)

        answ = model.predict(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        pred=str(round(answ[0][classification]*100 ,2)) + '% Memprediksi itu adalah ' + names(classification)   
        
        #Second prediction and facts about tumor if there is
        if classification==0:
            facts = 'Tumor Glioma adalah sejenis tumor yang terjadi di otak dan sumsum tulang belakang.\
                    Glioma dapat mempengaruhi fungsi otak dan mengancam jiwa tergantung pada\
                    lokasi dan tingkat pertumbuhannya.'
            no_tumor = str(round(answ[0][1]*100 ,2))
            pred2 = no_tumor + '% Memprediksi tidak ada tumor'
            
        elif classification==2:
            facts = 'Meningioma adalah tumor yang muncul dari meningen, selaput yang mengelilingi otak.\
                    Kebanyakan meningioma tumbuh sangat lambat, seringkali selama bertahun-tahun tanpa menimbulkan gejala.'
            no_tumor = str(round(answ[0][1]*100 ,2))
            pred2 = no_tumor + '% Memprediksi tidak ada tumor'
        
        elif classification==3:
            facts = 'Tumor Pituitary adalah pertumbuhan abnormal yang berkembang di kelenjar Pituitary.\
                    Kebanyakan tumor Pituitary adalah pertumbuhan non-kanker (jinak) yang tetap berada di Pituitary \
                    kelenjar atau jaringan sekitarnya.'
            no_tumor = str(round(answ[0][1]*100 ,2))
            pred2 = no_tumor + '% Memprediksi tidak ada tumor'
        
        else:
            facts=None
            pred2 = None
        
        return children, pred, pred2, facts
    
    else:
        return (no_update, no_update, no_update, no_update)  

if __name__ == '__main__':
    app.run_server(debug=True)
