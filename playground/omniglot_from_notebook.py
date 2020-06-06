import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch import tensor
from torch.functional import F
from torch.utils.data.dataset import TensorDataset

# import matplotlib.style as style
# print(style.available)
# style.use('seaborn-poster')  # sets the size of the charts
# style.use('ggplot')
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.rcParams['font.family'] = "serif"
plt.rcParams['figure.figsize'] = [14, 9]

# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# device = torch.device("cuda:0")
# !nvidia - smi
# # if no CPU available, one can switch to CPU ofc. Note that google colab offers free GPU power
device = torch.device("cpu")

# from google.colab import drive
# drive.mount("/gdrive/")
# datasets_location="/gdrive/My Drive/playground/datasets/"

# !ls -hail "/gdrive/My Drive/playground"
# !ls -hail "$datasets_location"

### TODO had problems with too slow loading of omniglot from drive :/

# or alternatively if one does not want to connect to google drive
datasets_location = "./datasets/"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

omniglot_background_dataset = torchvision.datasets.Omniglot(
    root=datasets_location,
    download=True,
    background=True,
    transform=transforms
)


def k_samples_from_each_class_into_train_rest_into_test(dataset, k):
    class_count = {}
    train_data_arr = []
    train_label_arr = []
    test_data_arr = []
    test_label_arr = []
    for image, label in dataset:
        class_count[label] = class_count.get(label, 0) + 1
        if class_count[label] <= k:
            train_data_arr.append(image.unsqueeze_(0))
            train_label_arr.append(tensor([label]))
        else:
            test_data_arr.append(image.unsqueeze_(0))
            test_label_arr.append(tensor([label]))
        # if class_count[label] == 20:
        #     break;

    train_data_arr = torch.cat(train_data_arr)
    train_label_arr = torch.cat(train_label_arr)
    test_data_arr = torch.cat(test_data_arr)
    test_label_arr = torch.cat(test_label_arr)

    return (TensorDataset(train_data_arr, train_label_arr),
            TensorDataset(test_data_arr, test_label_arr))


omniglot_train_dataset, omniglot_test_dataset = k_samples_from_each_class_into_train_rest_into_test(
    omniglot_background_dataset,
    20 * 80 // 100
)

omniglot_train_loader = torch.utils.data.DataLoader(
    omniglot_train_dataset,
    batch_size=720,
    shuffle=True
)

omniglot_test_loader = torch.utils.data.DataLoader(
    omniglot_test_dataset,
    batch_size=720,
    shuffle=True
)

# omniglot_evaluation_dataset = torchvision.datasets.Omniglot(
#     root=datasets_location,
#     download=True,
#     background=False,
#     transform=transforms
# )

# omniglot_evaluation_loader = torch.utils.data.DataLoader(
#     omniglot_evaluation_dataset,
#     batch_size = 72,
#     shuffle=True
# )

# PLAYGROUND

print("Background:", omniglot_background_dataset)
# print("Evaluation:", omniglot_evaluation_dataset)

image, label = omniglot_background_dataset[0]
print(type(image))
print(image.shape)
print(type(label))

images, labels = next(iter(omniglot_train_loader))
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(18, 18))
plt.axis('off')
plt.imshow(np.transpose(grid, (1, 2, 0)))

print(labels)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 6, 1)
        self.conv2 = nn.Conv2d(10, 14, 5, 1)
        self.fc1 = nn.Linear(in_features=14 * 23 * 23, out_features=5000)
        self.fc2 = nn.Linear(in_features=5000, out_features=964)

    def forward(self, x):
        # x is 1x105x105
        x = F.relu(self.conv1(x))  # 10x100x100
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 10x50x50

        x = F.relu(self.conv2(x))  # 14x46x46
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 14x23x23

        x = x.view(-1, 14 * 23 * 23)
        x = F.relu(self.fc1(x))  # 5000

        x = self.fc2(x)  # 964
        return F.log_softmax(x, dim=1)


class CnnKoch2015(nn.Module):
    def __init__(self):
        super(CnnKoch2015, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10, 1, )
        self.conv2 = nn.Conv2d(64, 128, 7, 1)
        self.conv3 = nn.Conv2d(128, 128, 4, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 1)

        self._init_convolution_layers(
            [self.conv1, self.conv2, self.conv3, self.conv4]
        )

        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=964)
        self._init_linear_layers([self.fc1, self.fc2])

    def _init_convolution_layers(self, conv_layers):
        for c in conv_layers:
            torch.nn.init.normal(c.weight, mean=0, std=0.01)
            torch.nn.init.normal(c.bias, mean=0.5, std=0.01)

    def _init_linear_layers(self, linear_layers):
        for l in linear_layers:
            torch.nn.init.normal(l.weight, mean=0, std=0.2)
            torch.nn.init.normal(l.bias, mean=0.5, std=0.01)

    def forward(self, x):
        # x is 1x105x105
        x = F.relu(self.conv1(x))  # 64x96x96
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 64x48x48

        x = F.relu(self.conv2(x))  # 128x42x42
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 128x21x21

        x = F.relu(self.conv3(x))  # 128x18x18
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 128x9x9

        x = F.relu(self.conv4(x))  # 256x6x6
        x = x.view(-1, 256 * 6 * 6)

        x = F.relu(self.fc1(x))  # 4096

        x = self.fc2(x)  # 964
        return F.log_softmax(x, dim=1)


# def train(model, device, train_loader, optimizer,
#           epochs, loss_fun, log_interval=72):
#     model.train()  # switch to train mode

#     loss_values = []

#     for e in range(1, epochs+1):
#         for batch_id, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)

#             output = model(data)
#             loss = loss_fun(output, target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loss_values.append(loss.item())
#             # !nvidia-smi

#             if batch_id%log_interval == 0:
#                 print("epoch{}/{} [{}/{} {:.0f}%], loss={}]".format(e, epochs,
#                       batch_id*len(data), len(train_loader.dataset),
#                       batch_id*100./len(train_loader), loss.item()))

#     print("done")
#     return loss_values

def train(model, device, train_loader, optimizer, epochs, loss_fun, test, test_loader, log_interval=72):
    # test and test_loader are used to log validation loss. TODO i think that train should not take test_loader. refactor
    model.train()  # switch to train mode

    # TODO refactor loss logging
    running_loss = 0.0
    loss_values = []
    validation_loss_values = []
    validation_accuracy_values = []

    i = 1
    for e in range(1, epochs + 1):
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_fun(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_values.append([i, loss.item() / len(data)])

            if batch_id % log_interval == 0:
                print("epoch{}/{} [{}/{} {:.0f}%], loss={}]".format(e, epochs,
                                                                    batch_id * len(data), len(train_loader.dataset),
                                                                    batch_id * 100. / len(train_loader), running_loss))
                running_loss = 0.0
                validation_result = test(model, device, test_loader, loss_fun)
                validation_loss_values.append([i, validation_result[0]])
                validation_accuracy_values.append([i, validation_result[1]])
            i += 1

    print("done")
    return [loss_values, validation_loss_values, validation_accuracy_values]


# def test(model, device, loader, loss_fun):
#     model.eval()

#     with torch.no_grad():
#         correct = 0
#         loss = 0.0
#         for data, target in loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)  # why keepdim?
#             loss += loss_fun(output, target).item()
#             correct += (pred.squeeze()==target).sum()  # TODO optimize
#             # correct += pred.eq(target.view_as(pred)).sum().item()

#         loss = loss/len(loader.dataset)
#         print("Test phase:\n\tAverage loss: {:.5f}\n\tAccuracy: {}/{} ({:.1f}%)".format(
#             loss, correct, len(loader.dataset), correct*100./len(loader.dataset)))

def test(model, device, loader, loss_fun):
    model.eval()

    with torch.no_grad():
        correct = 0
        loss = 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # why keepdim?
            loss += loss_fun(output, target).item() / len(data)
            correct += (pred.squeeze() == target).sum()  # TODO optimize
            # correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct * 100. / len(loader.dataset)
        loss = loss / len(loader)
        print("Test phase:\n\tAverage loss: {:.5f}\n\tAccuracy: {}/{} ({:.1f}%)".format(
            loss, correct, len(loader.dataset), accuracy))
        return [loss, accuracy]


torch.cuda.empty_cache()

net = CnnKoch2015().to(device)
print(net)

# training
epochs = 20
loss_fun = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(net.parameters(), lr=0.01)
[loss_values, validation_loss_values, validation_accuracy_values] = train(net, device, omniglot_train_loader, optimizer,
                                                                          epochs,
                                                                          loss_fun, test, omniglot_test_loader)

# #plotting loss
from scipy.ndimage.filters import gaussian_filter1d

fig, ax1 = plt.subplots(figsize=(18, 9))

color = 'orange'
ax1.set_xlabel('batch number')
ax1.set_ylabel('train batch loss', color=color)
ax1.plot(np.array(loss_values).T[0], np.array(loss_values).T[1], c="red", label="raw train loss")
ax1.plot(np.array(loss_values).T[0], gaussian_filter1d(np.array(loss_values).T[1], sigma=10), c=color,
         label="smooth train loss")
ax1.plot(np.array(validation_loss_values).T[0], np.array(validation_loss_values).T[1], c="yellow",
         label="validation loss")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = "green"
ax2.set_ylabel('validation accuracy', color=color)
ax2.plot(np.array(validation_accuracy_values).T[0], np.array(validation_accuracy_values).T[1], c=color,
         label="validation loss")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

# testing
test(net, device, omniglot_test_loader, loss_fun)
