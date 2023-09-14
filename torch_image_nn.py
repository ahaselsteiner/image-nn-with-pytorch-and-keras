# Neural net image classifier for digits using PyTorch.
#
# The NN has the same architecture as the Keras TensorFlow NN in keras_image_nn.py
#
# This code was adapted from this tutorial, https://www.youtube.com/watch?v=mozBidd58VQ
# Compared to the tutorial, the neural network architecture was
# changed: To speed up training, the second conv2D layer has only 32 outputs
# and maxpool layers were introduced.

import matplotlib.pyplot as plt
from time import perf_counter
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


do_train = False  # Do you want to train the model or only apply it?
n_epochs_to_train = 3


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        # Thanks to: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# Instance the neural network, loss, and optimizer
compute_type = "cpu"  # 'cuda' if you have GPU, otherwise 'cpu'
clf = ImageClassifier().to(compute_type)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


if __name__ == "__main__":
    model_file_name = "torch_model.pt"

    # Training
    if do_train:
        train = datasets.MNIST(
            root="data", download=True, train=True, transform=ToTensor()
        )
        dataset = DataLoader(train, 32)
        print(f"Traing a {clf.count_parameters()} parameter model: {clf}")
        for epoch in range(n_epochs_to_train):
            t1_start = perf_counter()
            for batch in dataset:
                X, y = batch
                X, y = X.to(compute_type), y.to(compute_type)
                yhat = clf(X)
                loss = loss_fn(yhat, y)

                # Apply backprop
                opt.zero_grad()
                loss.backward()
                opt.step()
            t1_end = perf_counter()
            print(
                f"Epoch: {epoch} loss is {loss.item()}, computing time was {t1_end - t1_start:.2f} seconds."
            )

        with open(model_file_name, "wb") as f:
            save(clf.state_dict(), f)

    # Prediction
    with open(model_file_name, "rb") as f:
        clf.load_state_dict(load(f))

    print(f"Predicting with a {clf.count_parameters()} parameter model: {clf}")
    img_files = ["img_1.jpg", "img_2.jpg", "img_3.jpg"]

    fig = plt.figure(figsize=(5, 4))
    for i, fname in enumerate(img_files):
        img = Image.open(fname)
        img_tensor = ToTensor()(img).unsqueeze(0).to(compute_type)
        predicted_number = torch.argmax(clf(img_tensor))
        ax = fig.add_subplot(1, len(img_files), i + 1, xticks=[], yticks=[])
        ax.imshow(img, cmap="gray")
        ax.set_title(str(int(predicted_number)))
    fig.suptitle("Predicted numbers")
    plt.show()
