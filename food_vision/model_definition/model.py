import tensorflow as tf
from food_vision.config import config
from food_vision.utils.callback_utils import (
    early_stop_callback,
    learning_rate_scheduler_callback,
    model_checkpoint_callback,
    tensorboard_callback,
)
from tensorflow.keras import layers


class Model:
    """
    Builds the modified efficient net model as a sklearn pipeline step
    """

    def __init__(
        self,
        input_shape=(224, 224, 3),
        output_shape=101,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def modified_EfficientNet_model(self):
        """Function to build the transfer learning model using EfficientNetB0 as a backbone"""
        # create base model
        base_model = tf.keras.applications.EfficientNetB4(include_top=False)
        base_model.trainable = False  # freeze base model layers

        # create functional inputs
        inputs = layers.Input(shape=self.input_shape, name="input_layer", dtype=tf.float32)
        a = base_model(inputs, training=False)
        b = layers.GlobalAveragePooling2D(name="average_pool")(a)
        c = layers.Dense(self.output_shape)(b)
        outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(c)

        model = tf.keras.Model(inputs, outputs)

        # compile the model
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            metrics=["accuracy"],
        )
        return model

    def train(self, training_data, test_data, experiment_name="food_vision_1"):
        """Method that trains the model with different settings"""

        # build the model
        model = self.modified_EfficientNet_model()

        # return training history
        training_history = model.fit(
            training_data,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(training_data),
            validation_data=test_data,
            validation_steps=int(0.15 * len(test_data)),
            callbacks=[
                tensorboard_callback("training_logs", experiment_name),
                model_checkpoint_callback(config.MODEL_PATH),
                early_stop_callback(monitor="val_accuracy", patience=2),
                learning_rate_scheduler_callback(
                    monitor="val_loss",
                    factor=0.2,
                    patience=2,
                    verbose=1,
                ),
            ],
        )

        return training_history

    def fine_tuning(self, training_data, test_data, experiment_name="food_vision_fine_tuned_1"):
        """ " Method that provides the fine tuning for the model"""
        # load the previously trained model
        try:
            trained_model = tf.keras.models.load_model(config.MODEL_PATH)
        except tf.errors.NotFoundError:
            print("Model not trained, please train the model first before fine-tuning")
            return

        # set all layers of the model to be trainable
        for layer in trained_model.layers:
            layer.trainable = True  # set all layers to trainable

        # recompile with a smaller learning rate
        trained_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(
                config.LEARNING_RATE / 10
            ),  # 10x lower learning rate than the default
            metrics=["accuracy"],
        )

        fine_tuned_history = trained_model.fit(
            training_data,
            epochs=5,  # fine-tune for a maximum of 100 epochs
            steps_per_epoch=len(training_data),
            validation_data=test_data,
            validation_steps=int(
                0.15 * len(test_data)
            ),  # validation during training on 15% of test data
            callbacks=[
                tensorboard_callback("training_logs", experiment_name),
                model_checkpoint_callback(config.FINE_TUNED_MODEL_PATH),
                early_stop_callback(monitor="val_accuracy", patience=2),
                learning_rate_scheduler_callback(
                    monitor="val_loss",
                    factor=0.2,
                    patience=2,
                    verbose=1,
                ),
            ],
        )

        return fine_tuned_history
