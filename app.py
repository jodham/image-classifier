from flask import Flask, request, render_template
import tensorflow as tf
from keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and set class names
model = load_model("my_image_classifier.keras")
img_size = (224, 224)
class_names = ['devotional', 'landscape', 'nature', 'people', 'flowers', 'animals']

def classify_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[tf.argmax(score)]
    confidence = 100 * tf.reduce_max(score)
    return predicted_class, confidence.numpy()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence = classify_image(filepath)
            return render_template('index.html', label=label, confidence=confidence, filename=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
