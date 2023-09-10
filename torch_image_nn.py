# This code was created by following the YouTube tutorial on building a neural 
# network with PyTorch, https://www.youtube.com/watch?v=mozBidd58VQ

import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)
        )
    

    def forward(self, x):
        return self.model(x)
    

# Instance the neural network, loss, and optimizer
compute_type = 'cpu' # 'cuda' if you have GPU, otherwise 'cpu'
clf = ImageClassifier().to(compute_type) 
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


do_train = False
if __name__ == "__main__":
    model_file_name = 'model_state.pt'

    # Training flow
    if do_train:
        for epoch in range(10):
            for batch in dataset:
                X, y = batch
                X, y = X.to(compute_type), y.to(compute_type)
                yhat = clf(X)
                loss = loss_fn(yhat, y)

                # Apply backprop
                opt.zero_grad()
                loss.backward()
                opt.step()

            print(f"Epoch:{epoch} loss is {loss.item()}")
        
        with open(model_file_name, 'wb') as f:
            save(clf.state_dict(), f)

    # Prediction flow
    with open(model_file_name, 'rb') as f:
        clf.load_state_dict(load(f))

    img_files = ['img_1.jpg', 'img_2.jpg', 'img_3.jpg']

    for fname in img_files:
        img = Image.open(fname)
        img_tensor = ToTensor()(img).unsqueeze(0).to(compute_type)
        print(f"{fname} is {torch.argmax(clf(img_tensor))}")
