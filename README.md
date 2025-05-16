# 🧠 Image Classifier Web App (Flask + TensorFlow)

A minimal web interface that lets you upload an image and classifies it using a pre-trained MobileNetV2 model.

## 🔧 Features
- Built using TensorFlow and Flask
- Upload any image via a browser
- Model predicts one of 6 custom classes
- Returns top prediction with confidence score

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/jodham/image-classifier.git
cd classifier

## Create virtual environment
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

## install dependencies
pip install -r requirements.txt

## run the app
python app.py

# creating dataset folders for your image files
create a parent folder name it "dataset"
-inside dataset create your different classes of folders
i.e people folder to hold images of people,
    animals to hold images of animals,
    houses to hold images of houses,
    flowers for flower images

# for errors on unrecognized file formats
run python singlefix.py 
in your terminal this will crosscheck all your image files in your dataset 
and fish out faulty or unrecognized file formats
## project structure

├── app.py                  # Flask web interface
├── my_image_classifier.keras  # Pretrained model
├── templates/index.html    # HTML interface
├── static/uploads/         # Uploaded images
├── requirements.txt
└── README.md
