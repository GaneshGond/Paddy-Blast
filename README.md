# Paddy Leaf Disease Diagnosis

This project is a web-based application for diagnosing diseases in paddy leaves. Users can upload an image of a leaf to receive a prediction and suggested solutions. The app is built with a visually appealing frontend and supports features like modal popups, error handling, and displaying additional analysis results such as heatmaps and confidence graphs.

---

## Features

- **Image Upload:** Allows users to upload images of paddy leaves.
- **Error Handling:** Displays an error message if no image is selected.
- **Disease Prediction:** Provides a diagnosis for the uploaded leaf image.
- **Heatmaps and Confidence Graphs:** Visual representations of model predictions.
- **Solutions Modal:** Displays suggested solutions for the detected disease.
- **Non-Paddy Leaf Alert:** Alerts users if the uploaded image is not of a paddy leaf.

---

## Project Structure

### Frontend
- **HTML**: Provides the structure of the web page.
- **CSS**: Ensures a clean and user-friendly design.
- **JavaScript**: Handles functionality, such as uploading images and displaying modals.

### Backend
- The backend includes an API endpoint `/upload` to process the uploaded images and return a JSON response with:
  - `prediction`: The diagnosis result.
  - `confidence`: The confidence score of the prediction.
  - `solution`: Suggested solutions for the detected disease.
  - `heatmap`: A URL to the heatmap image.
  - `confidence_graph`: A URL to the confidence graph.
  - `is_paddy`: Boolean to indicate if the uploaded image is a paddy leaf.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paddy-leaf-disease-diagnosis.git
   ```

2. Navigate to the project directory:
   ```bash
   cd paddy-leaf-disease-diagnosis
   ```

3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   python app.py
   ```
   - This will start the backend API, usually at `http://127.0.0.1:5000`.

5. Open a terminal to start a local server for the frontend. You can use Python to serve the static files:
   ```bash
   python -m http.server
   ```

6. Open your browser and navigate to `http://127.0.0.1:8000` (or the port displayed in your terminal).

---

## Usage

1. Open the application in a web browser.
2. Upload an image of a paddy leaf.
3. Click on the **Analyze** button to get the diagnosis.
4. View the results, including predictions, heatmaps, and suggested solutions.

---

## Technologies Used

- **Frontend:**
  - HTML
  - CSS
  - JavaScript

- **Backend:**
  - Flask (Python)
  - TensorFlow/Keras models for image classification

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- Inspired by the need to assist farmers in diagnosing leaf diseases effectively.
- Background image and icons sourced from open-source resources.

---

## Future Enhancements

- Add support for detecting multiple diseases in a single image.
- Implement backend functionality for handling image uploads and predictions.
- Improve the UI for mobile responsiveness.
- Enhance the model's accuracy and extend it to other crop types.

