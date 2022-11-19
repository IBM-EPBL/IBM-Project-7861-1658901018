import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)

model1 = load_model("fruit.h5")
model = load_model("vegetable.h5")
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        
    if (plant == "fruit"):    
        pred=np.argmax(model1.predict(x),axis=1)
        df = pd.read_excel('precautions - fruits.xlsx')
        print(df.iloc[pred[0]]['caution'])
    else:
        pred=np.argmax(model.predict(x),axis=1)
        df = pd.read_excel('precautions - veg.xlsx')
        print(df.iloc[pred[0]]['caution'])

    return df.iloc[pred[0]]['caution']
if __name__=='__main__':
    app.run(debug=False)
