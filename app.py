from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0 : 'Level 0',  
       1 : 'Level 1',
       2 : 'Level 2', 
       3 : 'Level 3'}

model = load_model('model/acnelevel-v4.h5')

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 224, 224, 3)
    predictions = model.predict(i)
    predicted_class_index = np.argmax(predictions)
    return dic[predicted_class_index]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
 return render_template("classification.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
 if request.method == 'POST':
  img = request.files['my_image']

  img_path = "static/" + img.filename 
  img.save(img_path)

  p = predict_label(img_path)

 return render_template("classification.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
 #app.debug = True
 app.run(debug = True)