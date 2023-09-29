import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

##Convert image into numbers and normalizes the tensor with a mean and standard deviation
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
##download the data sets and load them into dataloader
trainset = datasets.MNIST('/home/lingjia/Desktop/CSCE790/HW1/trainset1', download=True, train=True, transform=transform)
valset = datasets.MNIST('/home/lingjia/Desktop/CSCE790/HW1/testset1', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

##check out the shape of the iamges and the labels
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

#display one image from the training set
# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
# plt.show()

#figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
#     plt.show(figure)

##Build the Neural Network

# Layer details for the neural network
input_size = 784
firstl = 128 ##default:128
secl = 64   ##default:64
hidden_sizes = [firstl, secl] ##Lingjia: default:[128,64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), #the first layer, a linear (fully connected) layer that takes an input of size input_size and produces an output of size hidden_sizes[0]. It represents an affine transformation of the input data.
                      nn.ReLU(), ##apply a rectified linear unit (ReLU) activation function. This introduces non-linearity into the model by applying the function max(0, x) to each element of the tensor produced by the previous linear layer.
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]), ##This is the second linear layer, which takes the output of the first hidden layer (hidden_sizes[0]) and produces an output of size hidden_sizes[1].
                      nn.ReLU(),  ##apply another ReLU activation function.
                      nn.Linear(hidden_sizes[1], output_size), ## the final linear layer, which takes the output of the second hidden layer (hidden_sizes[1]) and produces the final output of size output_size.
                      nn.LogSoftmax(dim=1)) ##apply a log softmax activation along dimension 1 (usually used for classification tasks with multiple classes). This converts the raw scores produced by the last linear layer into log probabilities.
print(model)

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

##Define the negative log-likelihood loss (NLLLoss())
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images.cuda()) #log probabilities
loss = criterion(logps, labels.cuda()) #calculate the NLL loss

##adjusting weights
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images.cuda())
loss = criterion(output, labels.cuda())
loss.backward()
print('Gradient -', model[0].weight.grad)
#
# ##Core training Process
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images.cuda())
        loss = criterion(output, labels.cuda())

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

images, labels = next(iter(valloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img.cuda())

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.cpu().numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)

correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img.cuda())

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.cpu().numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)

print(f"\nModel Accuracy with {firstl} neurons in Hidden Layer 1 and {secl} neurons in Hidden Layer 2 =", (correct_count/all_count))


#torch.save(model, './my_mnist_model.pt')