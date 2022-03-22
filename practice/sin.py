import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

input_number = 1
hidden_number = 30
output_number = 1
epoch = 10000
batch_size = 10000
learning_rate = 0.001
test_epoch = 1000
min_value = -12.56
max_value = 12.56

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinearModel(nn.Module):
    def __init__(self, input_number, hidden_number, output_number):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_number, hidden_number)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_number, output_number)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = LinearModel(input_number, hidden_number, output_number).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epoch):
    x_train = [random.uniform(min_value, max_value) for x in range(batch_size)]
    y_train = [math.sin(x) for x in x_train]
    x_train = torch.tensor(x_train).reshape(-1, 1)
    y_train = torch.tensor(y_train).reshape(-1, 1)
    output = model(x_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 50 == 0:
        print("Epoch: [{}/{}] train complete, Loss: {:.4f}".format(i + 1, epoch, loss.item()))

torch.save(model.state_dict(), "sin_model.ckpt")

with torch.no_grad():
    correct = 0
    x_test = [random.uniform(min_value, max_value) for x in range(test_epoch)]
    y_test = [math.sin(x) for x in x_test]
    x_test = torch.tensor(x_test).reshape(-1, 1)
    y_test = torch.tensor(y_test).reshape(-1, 1)
    outputs = model(x_test)
    plt.plot(x_test, outputs, 'ro', label='Original data')
    plt.legend()
    plt.show()
    for i in range(len(outputs)):
        if math.fabs(outputs[i] - y_test[i]) <= 1e-2:
            correct += 1
    print('Accuracy of the network on the {} test {} %'
          .format(test_epoch, 100 * correct / test_epoch))
input = input("******")
