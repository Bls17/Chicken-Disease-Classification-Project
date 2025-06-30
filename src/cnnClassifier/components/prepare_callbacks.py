import os
import time
import tensorflow as tf
from src.cnnClassifier.config.configuration import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            str(self.config.tensorboard_root_log_dir),
            f"tb_logs_at_{timestamp}"
        )

        # Assure-toi que le dossier existe
        os.makedirs(tb_running_log_dir, exist_ok=True)

        return tf.keras.callbacks.TensorBoard(
            log_dir=tb_running_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )

    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )

    def get_tb_ckpt_callbacks(self):
        return [
            # self._create_tb_callbacks(),  # Désactive temporairement
            self._create_ckpt_callbacks()
        ]

