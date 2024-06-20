import cv2
import pickle
class VideoCamera:
    def __init__(self):
        # Initialize your camera or other video capturing device
        self.video = cv2.VideoCapture(0)  # Use 0 for the default camera (you might need to change it based on your setup)
        self.model = pickle.load(open('model.p', 'rb'))


    def __del__(self):
        # Release the video capturing device when the object is destroyed
        self.video.release()

    def predict_sign_language(self, frame):
        # Perform any necessary preprocessing on the frame (resize, normalization, etc.)
        # Here, we'll resize the frame to match the expected input size of your model
        resized_frame = cv2.resize(frame, (50, 50))

        # Perform inference using your pre-trained model
        # You need to replace this line with the actual prediction logic based on your model architecture
        prediction = self.model.predict(resized_frame.reshape(1, -1))

        # Return the predicted sign or label
        return prediction

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        # Get sign language prediction for the current frame
        sign_prediction = self.predict_sign_language(frame)

        # Draw the prediction on the frame
        cv2.putText(frame, f'Predicted Sign: {sign_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert the frame to bytes for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()