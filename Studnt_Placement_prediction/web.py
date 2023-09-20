from flask import Flask,render_template,request
import pickle
import numpy as np 
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD FOLDER'] = 'static'

model = pickle.load(open('mnist.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predi', methods=['POST','GET'])
def predi():
    cgpa = request.form['cgpa']
    iq = request.form['iq']
    print(cgpa)
    print(iq)
    res = pd.DataFrame({'cgpa': [cgpa], 'iq': [iq]})
    # res = {'cgpa':cgpa,'iq':iq}
    pre = model.predict(res)
    print(pre)
    return render_template('result.html',pre = pre)
@app.route('/mnist', methods=['POST','GET'])
def mnist():
    img = request.files['image']
    img.save(os.path.join(app.config['UPLOAD FOLDER'],img.filename))
    y_p=model.predict(img)
    y_prod = y_p.argmax(axis=1)
    return render_template('result.html',y_prod = y_prod)
if __name__ == "__main__":
    app.run(debug=True,port=8000)