# Neural net image classifier for digits using Keras with TensorFlor.
#
# The NN has the same architecture as the PyTorch NN in torch_image_nn.py
#
# This medium post helped me write the code: https://medium.com/p/555007a50c2e

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from PIL import Image

import tensorflow as tf
from keras.models import load_model
from keras.utils import plot_model

do_train = False # Do you want to train the model or only apply it?
n_epochs_to_train = 3

model_fname = "keras_model.h5"


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Training
    if do_train:
        fig = plt.figure(figsize=(5, 4))
        for idx in np.arange(10):
            ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
            ax.imshow(x_train[idx].reshape(28, 28), cmap="gray")
            ax.set_title(str(y_train[idx].item()))
        plt.suptitle("MINST training dataset (first 10 images)")
        plt.show()

        """
        Model in https://medium.com/p/555007a50c2e (maxPool makes training faster)
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, (5, 5), activation=tf.nn.relu)(inputs)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (5, 5), activation=tf.nn.relu)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(120, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(84, activation=tf.nn.relu)(x)
        """

        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu)(inputs)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)

        # Keras Tensorflow .Dense is similar as PyTorch nn.Linear if the function's
        # parameters are,
        # see https://stackoverflow.com/questions/72803869/are-those-keras-and-pytorch-snippets-equivalent
        x = tf.keras.layers.Dense(10, activation=None)(x)
        """
        The neural net in PyTorch:
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )
        """

        outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)
        model = tf.keras.Model(inputs, outputs)
        model.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_fname,
            monitor="accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        t1_start = perf_counter()
        model.fit(
            x_train,
            y_train,
            epochs=n_epochs_to_train,
            verbose=2,
            callbacks=[checkpoint],
        )
        t1_end = perf_counter()
        print(f"Time for training using Keras is {t1_end - t1_start:.2f} seconds.")

    model = load_model(model_fname)

    # Plotting the model requires the graphviz package which is difficult to install
    # plot_model(model, to_file='keras_model_plot.png', show_shapes=True, show_layer_names=True)

    # Prediction
    print(
        f"Predicting with a {model.count_params()} parameter model: {model.summary()}"
    )
    t1 = perf_counter()
    test_loss, test_acc = model.evaluate(x_test, y_test)
    t2 = perf_counter()
    print(
        f"Evaluation accuracy using Keras is {test_acc * 100:.2f} and execution time {t2 - t1:.2f} seconds."
    )

    img_files = ["img_1.jpg", "img_2.jpg", "img_3.jpg"]

    fig = plt.figure(figsize=(5, 4))
    for i, fname in enumerate(img_files):
        img = Image.open(fname)
        img_tensor = tf.convert_to_tensor(img) / 255

        img_tensor = np.expand_dims(img_tensor, axis=2)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        predictions = model.predict(img_tensor)
        predicted_number = np.argmax(predictions)
        ax = fig.add_subplot(1, len(img_files), i + 1, xticks=[], yticks=[])
        ax.imshow(img, cmap="gray")
        ax.set_title(str(predicted_number))
    fig.suptitle("Predicted numbers")
    plt.show()
