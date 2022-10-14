
import tensorflow as tf
import time

class Cleaner():
    def __init__(self, model_type="unet", batch_size=64):
        self.model = None
        self.model_type = model_type
        self.batch_size = batch_size

    def load(self, model_path: str):
        start_time = time.time()
        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"PSNR": None, "SSIM": None})
        self.model_load_time = time.time() - start_time

    def clean(self, image_np):
        return self.model.predict(image_np, batch_size=self.batch_size)

    def is_loaded(self):
        return self.model is not None
