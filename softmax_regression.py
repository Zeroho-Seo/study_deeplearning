import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import torch.nn as nn


torch.manual_seed(1)

#소프트맥스 회귀 구현하기
'''x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]] #[8,4]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)


y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

W=torch.zeros((4,3), requires_grad= True)
b = torch.zeros(1, requires_grad= True)
optimizer = optim.SGD([W,b], lr= 0.01)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)
    # hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    # cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))'''

#소프트맥스 회귀 MNIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

training_epochs = 15
batch_size = 100

mnist_train = dset.MNIST(root = './FashionMNIST',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
mnist_test = dset.MNIST(root = './FashionMNIST',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle = True,
                         drop_last=True)
#sampling
'''sample_idx = torch.randint(len(mnist_train), size=(1,)).item()
img, label = mnist_train[sample_idx+1]
plt.plot()
plt.imshow(img.squeeze(), cmap="gray")
plt.show()'''

linear = nn.Linear(784, 10, bias=True).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(linear.parameters(), lr =0.01)

for epoch in range(training_epochs):
    avg_cost=0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)
        Y= Y.to(device)
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)


        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')


with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
    print("Label: ", Y_single_data.item())
    single_prediction = linear(X_single_data)
    print("Prediction: ", torch.argmax(single_prediction, 1).item())
    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()