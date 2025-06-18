from ultralytics import YOLO
from PIL import Image

tuned_model = YOLO(r"runs/train_run_1/weights/best.onnx")


def predict_image(img):
    result = tuned_model.predict(
        img
    )
    
    result = result[0]
    
    img = result.plot()
    
    pil_image = Image.fromarray(img)
    
    
    return pil_image

if __name__ == "__main__":
    predicted_image = predict_image(r"safety-helmet-1/valid/images/71yhwNnc5OL_jpg.rf.224ee60627e578de5a2a1d9af98c04ef.jpg")
    predicted_image.show()