import config
import tensorflow as tf
import tensorflow_datasets as tfds
from food_vision.model_definition.model import Model
from food_vision.utils.preprocessing_utils import preprocess_image


def run_training(save_result: bool = True):

    # get training data from tensorflow datasets
    (train_data, test_data), ds_info = tfds.load(
        name="food101",  # target dataset to get from TFDS
        split=[
            "train",
            "validation",
        ],  # what splits of data should we get? note: not all datasets have train, valid, test
        shuffle_files=True,  # shuffle files on download?
        as_supervised=True,  # download data in tuple format (sample, label), e.g. (image, label)
        with_info=True,
    )  # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

    train_data = train_data.map(
        map_func=lambda img, label: preprocess_image(img, label, img_shape=224, scale=False),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_data = (
        train_data.shuffle(buffer_size=5000)
        .batch(batch_size=32)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_data = test_data.map(
        map_func=lambda img, label: preprocess_image(img, label, img_shape=224, scale=False),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_data = (
        test_data.shuffle(buffer_size=5000)
        .batch(batch_size=32)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # define model and run training
    model = Model(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3), output_shape=config.NUM_CLASSES
    )
    model.train(train_data, test_data)


if __name__ == "__main__":
    run_training(save_result=True)
