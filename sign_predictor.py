import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import cv2  
import numpy as np
from keras.models import load_model
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    # Set the properties for the speech
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.8)

    # Speak the text
    engine.say(text)
    engine.runAndWait()


np.set_printoptions(suppress=True)
# Load the model
model = load_model("keras_Model.h5", compile=False)
# Load the labels
class_names = open("labels.txt", "r").readlines()

def predict(image):
    
    # resize the image to the models input shape
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    return {
        "class": class_name[2:],
        "confidence": str(np.round(confidence_score * 100))[:-2]
    }


def detect_sign():
    cap = cv2.VideoCapture(0)
    print('cap', cap)
    detector = HandDetector(maxHands=1)
    print('detector', detector)
    offset = 20
    imgSize = 300
    print("Press q to quit")
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            result = predict(imgWhite)
            out = result['class']
            cnf = result['confidence']
            img = cv2.putText(img, out, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            img = cv2.putText(img, cnf, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f"hand", imgWhite)
            speak(result['class'])
        cv2.imshow("output", img)

        if cv2.waitKey(1) == ord("q"):
            break   
    cv2.destroyAllWindows()
    cap.release()



if __name__ == "__main__":
    detect_sign()