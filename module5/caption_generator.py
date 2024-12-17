import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.applications import DenseNet201

TOKENIZER = None
with io.open('tokenizer.json', 'r') as file:
    TOKENIZER = tokenizer_from_json(file.read())

CAPTION_MODEL = load_model('model.h5')

model = DenseNet201()
features_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)


def get_features(image_path):
    img = Image.open(io.BytesIO(image_path))
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    feature = features_extractor.predict(img, verbose=0)
    return feature

def idx_to_word(integer):
    for word, index in TOKENIZER.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(max_length, feature):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = TOKENIZER.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = CAPTION_MODEL.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text.replace('startseq', '').replace('endseq', '').strip()

def predict(image_path):
    feature = get_features(image_path)
    caption = predict_caption(34, feature)
    return caption
