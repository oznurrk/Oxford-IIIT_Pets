import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
IMG_SIZE = 224

# Modeli yükle
model = load_model('oxford_pets_model.h5')

# Oxford Pets 37 sınıfı
class_names = [
    'Abyssinian', 'American_bulldog', 'American_pit_bull_terrier', 'Basset_hound',
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_Shorthair', 'Chihuahua',
    'Egyptian_Mau', 'English_cocker_spaniel', 'English_setter', 'German_shorthaired',
    'Great_Pyrenees', 'Havanese', 'Japanese_chin', 'Keeshond', 'Leonberger', 'Maine_Coon',
    'Miniature_pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll',
    'Russian_Blue', 'Saint_Bernard', 'Samoyed', 'Scottish_Terrier', 'Shiba_Inu', 'Siamese',
    'Sphynx', 'Staffordshire_bull_terrier', 'Wheaten_terrier', 'Yorkshire_terrier'
]

# app.py içinde ilgili kısmı değiştiriyoruz
CONFIDENCE_THRESHOLD = 60  # %60 altında olanlar için 'bulunamadı' yaz

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_url = filepath

            # Görseli hazırla
            img = Image.open(filepath).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Tahmin
            preds = model.predict(img_array)
            class_id = np.argmax(preds[0])
            confidence = np.max(preds[0]) * 100

            if confidence < CONFIDENCE_THRESHOLD:
                prediction = "Hayvan türü bulunamadı. (%{:.2f})".format(confidence)
            else:
                prediction = f"{class_names[class_id]} ({confidence:.2f}%)"

    return render_template('index.html', prediction=prediction, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
