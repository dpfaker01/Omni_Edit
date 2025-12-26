import os
from PIL import Image
import numpy as np
import onnxruntime as ort
import json
from huggingface_hub import hf_hub_download


class NSFWDetector:
    """
    NSFW detector using YOLOv9 for image classification.
    """
    
    def __init__(self, repo_id="Falconsai/nsfw_image_detection", 
                 model_filename="falconsai_yolov9_nsfw_model_quantized.pt",
                 labels_filename="labels.json",
                 input_size=(224, 224)):
        """
        Initialize the NSFW detector.
        
        Args:
            repo_id (str): Hugging Face repository ID.
            model_filename (str): Model filename.
            labels_filename (str): Labels filename.
            input_size (tuple): Model input size (height, width).
        """
        self.repo_id = repo_id
        self.model_filename = model_filename
        self.labels_filename = labels_filename
        self.input_size = input_size
        
        # Download files from Hugging Face
        self.model_path = self._download_model()
        self.labels_path = self._download_labels()
        
        # Load labels
        self.labels = self._load_labels()
        
        # Load model
        self.session = self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def _download_model(self):
        """
        Download the model file from Hugging Face.
        
        Returns:
            str: Path to the downloaded model file.
        """
        try:
            print(f"Downloading model from {self.repo_id}: {self.model_filename}")
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.model_filename,
                cache_dir="./hf_cache"
            )
            print(f"✅ Model downloaded: {model_path}")
            return model_path
        except Exception as e:
            raise RuntimeError(f"Model download failed: {e}")
    
    def _download_labels(self):
        """
        Download the labels file from Hugging Face.
        
        Returns:
            str: Path to the downloaded labels file.
        """
        try:
            print(f"Downloading labels from {self.repo_id}: {self.labels_filename}")
            labels_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.labels_filename,
                cache_dir="./hf_cache"
            )
            print(f"✅ Labels downloaded: {labels_path}")
            return labels_path
        except Exception as e:
            raise RuntimeError(f"Labels download failed: {e}")
    
    def _load_labels(self):
        """
        Load class labels.
        
        Returns:
            dict: Labels dictionary.
        """
        try:
            with open(self.labels_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Labels file is malformed: {self.labels_path}")
    
    def _load_model(self):
        """
        Load ONNX model.
        
        Returns:
            onnxruntime.InferenceSession: Model session.
        """
        try:
            return ort.InferenceSession(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Model load failed: {self.model_path}, error: {e}")
    
    def _preprocess_image(self, image_path):
        """
        Preprocess image.
        
        Args:
            image_path (str): Image file path.
            
        Returns:
            tuple: (preprocessed tensor, original image)
        """
        try:
            # Load and convert image
            original_image = Image.open(image_path).convert("RGB")
            
            # Resize
            image_resized = original_image.resize(self.input_size, Image.Resampling.BILINEAR)
            
            # To numpy and normalize
            image_np = np.array(image_resized, dtype=np.float32) / 255.0
            
            # Reorder dims [H, W, C] -> [C, H, W]
            image_np = np.transpose(image_np, (2, 0, 1))
            
            # Add batch dim [C, H, W] -> [1, C, H, W]
            input_tensor = np.expand_dims(image_np, axis=0).astype(np.float32)
            
            return input_tensor, original_image
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {e}")
    
    def _postprocess_predictions(self, predictions):
        """
        Postprocess model predictions.
        
        Args:
            predictions: Model output.
            
        Returns:
            str: Predicted class label.
        """
        predicted_index = np.argmax(predictions)
        predicted_label = self.labels[str(predicted_index)]
        return predicted_label
    
    def predict(self, image_path):
        """
        Run NSFW detection on a single image.
        
        Args:
            image_path (str): Image file path.
            
        Returns:
            tuple: (predicted label, original image)
        """
        # Preprocess image
        input_tensor, original_image = self._preprocess_image(image_path)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        predictions = outputs[0]
        
        # Postprocess
        predicted_label = self._postprocess_predictions(predictions)
        
        return predicted_label, original_image
    
    def predict_label_only(self, image_path):
        """
        Return only the predicted label (no image).
        
        Args:
            image_path (str): Image file path.
            
        Returns:
            str: Predicted class label.
        """
        predicted_label, _ = self.predict(image_path)
        return predicted_label
    
    def predict_from_pil(self, pil_image):
        """
        Run NSFW detection from a PIL Image object.
        
        Args:
            pil_image (PIL.Image): PIL image object.
            
        Returns:
            tuple: (predicted label, original image)
        """
        try:
            # Ensure RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            # Resize
            image_resized = pil_image.resize(self.input_size, Image.Resampling.BILINEAR)
            
            # To numpy and normalize
            image_np = np.array(image_resized, dtype=np.float32) / 255.0
            
            # Reorder dims [H, W, C] -> [C, H, W]
            image_np = np.transpose(image_np, (2, 0, 1))
            
            # Add batch dim [C, H, W] -> [1, C, H, W]
            input_tensor = np.expand_dims(image_np, axis=0).astype(np.float32)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            predictions = outputs[0]
            
            # Postprocess
            predicted_label = self._postprocess_predictions(predictions)
            
            return predicted_label, pil_image
            
        except Exception as e:
            raise RuntimeError(f"PIL image prediction failed: {e}")
    
    def predict_pil_label_only(self, pil_image):
        """
        Return only the predicted label from a PIL Image.
        
        Args:
            pil_image (PIL.Image): PIL image object.
            
        Returns:
            str: Predicted class label.
        """
        predicted_label, _ = self.predict_from_pil(pil_image)
        return predicted_label

# --- Usage example ---
if __name__ == "__main__":
    # Config
    single_image_path = "datas/bad01.jpg"
    
    try:
        # Create detector (auto-download from Hugging Face)
        detector = NSFWDetector()
        
        # Check image file exists
        if os.path.exists(single_image_path):
            # Run prediction
            predicted_label = detector.predict_label_only(single_image_path)
            print(f"Image file: {single_image_path}")
            print(f"Prediction: {predicted_label}")
        else:
            print(f"Error: Image file does not exist: {single_image_path}")
            
    except Exception as e:
        print(f"Error initializing detector: {e}")