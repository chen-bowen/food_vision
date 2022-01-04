from datetime import datetime

import tensorflow as tf


def tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.

    Stores log files with the filepath:
      "dir_name/experiment_name/current_datetime/"

    Args:
      dir_name: target directory to store TensorBoard log files
      experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def model_checkpoint_callback(checkpoint_dir, save_weights_only=False):
    """
    Create a model save checkpoint given the saved directory

    Args:
      checkpoint_dir (str): target directory to store model checkpoint files
    """
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        montior="val_accuracy",  # save the model weights with best validation accuracy
        save_best_only=True,  # only save the best weights
        save_weights_only=save_weights_only,  # only save model weights (not whole model)
        verbose=1,
    )
    return model_checkpoint


def early_stop_callback(monitor="val_loss", patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor, patience)
    return early_stopping


def learning_rate_scheduler_callback(
    monitor="val_accuracy",
    factor=0.2,  # multiply the learning rate by 0.2 (reduce by 5x)
    patience=2,
    verbose=1,  # print out when learning rate goes down
    min_lr=1e-7,
):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor, factor, patience, verbose, min_lr=min_lr
    )
    return reduce_lr
