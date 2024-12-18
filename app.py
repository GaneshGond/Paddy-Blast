import os
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt

# Load the model
try:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'paddy_disease_diagnosis_model.h5')
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load the model. Error: {e}")

# Define constants
IMG_SIZE = (128, 128)
CLASS_LABELS = ['Healthy', 'Mildly Diseased', 'Moderately Diseased', 'Severely Diseased']

SOLUTIONS = {
    'Healthy': "No action required. The plant is healthy.",
    'Mildly Diseased': """
        Consider applying organic pesticides and monitoring the plant regularly. 
        You can purchase fungicides for mild disease treatment:
        <div>
            <button onclick="window.open('https://www.badikheti.com/fungicide/pdp/indofil-baan-tricyclazole-75-wp-fungicide/1gnsxjmk', '_blank')">
                Buy Fungicide
            </button>
        </div>
    """,
    'Moderately Diseased': """
        Apply appropriate chemical treatment and isolate the plant if possible. 
        You can buy recommended fungicides for moderate disease:
        <div>
            <button onclick="window.open('https://krishisevakendra.in/products/propiconazole-13-9-difenoconazole-13-9-ec-prodizole-fungicide?variant=45290939547944&country=IN&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&srsltid=AfmBOorehrZbvzuP_2IHcR70pFn-uh2yunk3eogQp2U3Vrmrd68RTgWggIo', '_blank')">
                Buy Fungicide
            </button>
        </div>
    """,
    'Severely Diseased': """
        Remove the infected leaves or plant to prevent spreading and apply intensive treatment. 
        Purchase fungicides for severe disease treatment:
        <div>
            <button onclick="window.open('https://www.kisanshop.in/en/product/shivalik-zoxitop-fungicide-azoxystrobin-18-2percent-difenoconazole-11-4percent-sc?srsltid=AfmBOope5bou52WUQxVgKUFUd7FmQoko0ewpUZtso1X09RqPv11abooTx7g', '_blank')">
                Buy Fungicide
            </button>
        </div>
    """
}

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_paddy_leaf(img_path):
    """Validate if the uploaded image is a paddy leaf."""
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        # Load a pre-trained binary classification model for paddy leaf validation
        LEAF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'paddy_leaf_validator_model.h5')
        leaf_model = load_model(LEAF_MODEL_PATH)

        # Predict if the image is a paddy leaf
        is_leaf_prediction = leaf_model.predict(img_array)[0][0]  # Assume binary output: 0 (not a leaf), 1 (leaf)
        return is_leaf_prediction > 0.5  # Return True if it's a paddy leaf
    except Exception as e:
        raise RuntimeError(f"Failed to validate the image as a paddy leaf. Error: {e}")

def predict_image(img_path):
    try:
        # Validate if the image is a paddy leaf
        if not is_paddy_leaf(img_path):
            return "Not a Paddy Leaf", 0.0, "The uploaded image is not recognized as a paddy leaf. Please upload a valid paddy leaf image.", "", ""

        # Proceed with disease prediction
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]
        predicted_class = CLASS_LABELS[class_index]
        solution = SOLUTIONS[predicted_class]

        heatmap_base64 = generate_contours_heatmap(img_path)
        confidence_graph_base64 = generate_confidence_graph(predictions[0])

        return predicted_class, confidence, solution, heatmap_base64, confidence_graph_base64
    except Exception as e:
        raise RuntimeError(f"Failed to process the image. Error: {e}")


def generate_confidence_graph(predictions):
    try:
        plt.figure(figsize=(6, 4))
        plt.bar(CLASS_LABELS, predictions, color='skyblue')
        plt.xlabel("Classes")
        plt.ylabel("Confidence")
        plt.title("Prediction Confidence")
        plt.ylim(0, 1)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return graph_base64
    except Exception as e:
        raise RuntimeError(f"Failed to generate the confidence graph. Error: {e}")

def generate_contours_heatmap(img_path):
    image_bgr = cv2.imread(img_path)
    image_bgr = cv2.resize(image_bgr, IMG_SIZE)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image_rgb.copy()
    cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 1)

    _, buffer = cv2.imencode('.png', image_with_contours)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

@app.route('/')
def home():
    solution = SOLUTIONS['Severely Diseased']
    return render_template('index.html', solution=solution)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            result, confidence, solution, heatmap_base64, confidence_graph_base64 = predict_image(file_path)
            return jsonify({
                'prediction': result,
                'confidence': f"{confidence:.2f}",
                'solution': solution,
                'heatmap': heatmap_base64,
                'confidence_graph': confidence_graph_base64,
                'is_paddy': 'false'
            })
        except Exception as e:
            return jsonify({'error': f"Prediction failed. Error: {e}"}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only .png, .jpg, .jpeg allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)