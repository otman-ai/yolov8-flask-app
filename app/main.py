from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO
app = Flask(__name__)

model = YOLO("app/yolov8n.pt")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file from the request
        uploaded_file = request.files['image']

        if uploaded_file is None or uploaded_file.filename != '':
            # Read the image data
            image_data = uploaded_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            prediction = model.predict(image_cv2, save=False)
            data = {}
            for r in prediction:
                for i, box in enumerate(r.boxes):
                    print(box.data.tolist())
                    data[str(i)] = {'class index':int(box.cls), 
                                    'class name':model.names[int(box.cls)],
                                    'boxes':box.data.tolist()[0][:4],
                                    'prob':box.data.tolist()[0][4]}

            return jsonify(data)
        elif not allowed_file(uploaded_file):
            return jsonify({'error': 'format not supported'})
        else:
            return jsonify({'error': 'No file provided'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='localhost', port=5000,debug=True)
