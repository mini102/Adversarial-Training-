from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
#       see, https://github.com/pytorch/vision/issues/3497 for more information

#####역전파 그라디언트를 기반으로 가중치를 조정하여 손실을 최소화하는 작업이 아니라 공격 이 동일한 역전파 그라디언트를 기반으로 손실을 최대화하도록 입력 데이터를 조정#########
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True

#########################목적: 모델과 데이터 로더를 정의한 다음 모델을 초기화하고 사전 훈련된 가중치를 로드하는 것#############################
# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):   #torch.nn.Module을 상속 __init__과 forward를 ovveride/ 모든 신경망 모델은 nn.Module의 subclass
    def __init__(self):   #__init()__에서는 모델에서 사용될 module(nn.Linear, nn.Conv2d), activation function(nn.functional.relu, nn.functional.sigmoid)등을 정의
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):   #델에서 실행되어야하는 계산을 정의/input을 넣어서 어떤 계산을 진행하하여 output이 나올지를 정의
        x = F.relu(self.conv1(x))   #F = nn.functions
        x = F.max_pool2d(x, kernel_size=2, stride=2)  #Pooling layer는 image를 downsampling하여 더 작고 manageable하게 만듦.
        x = F.relu(self.conv2(x))   #Non-linear activations
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        #Dropout: 한 레이어마다 진행, Forward pass 과정에서 임의로 일부 뉴런을 0으로 만드는 것.
        x = x.view(-1, 4 * 4 * 50) # [batch_size, 50, 4, 4]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)
notTrained  = Net().to(device)

criterion = torch.nn.CrossEntropyLoss()   #loss_Func
optimizer = optim.SGD(notTrained.parameters(), lr=0.01)

notTrained.train()  # 학습을 위함
for epoch in range(3):
  for index, (data, target) in enumerate(test_loader):   #index = batch (일괄 처리), data= input , target = 정답
    optimizer.zero_grad()  # 기울기 초기화
    output = notTrained(data.to(device))
    loss = criterion(output, target.to(device))
    loss.backward()  # 역전파
    optimizer.step()   #역전파 단계에서 수집된 변화도로 매개변수를 조정

    if index % 2000 == 0:
      print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))

notTrained.eval()
# Load the pretrained model
#model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

criterion = torch.nn.CrossEntropyLoss()   #loss_Func
optimizery = optim.SGD(model.parameters(), lr=0.01)

model.train()  # 학습을 위함
for epoch in range(3):
  for index, (data, target) in enumerate(test_loader):   #index = batch (일괄 처리), data= input , target = 정답
    optimizery.zero_grad()  # 기울기 초기화
    output = model(data.to(device))
    loss = criterion(output, target.to(device))
    loss.backward()  # 역전파
    optimizery.step()   #역전파 단계에서 수집된 변화도로 매개변수를 조정

    if index % 2000 == 0:
      print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()


# FGSM attack code
#함수는 다음과 같이 교란된 이미지를 생성
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def train(model, data, target):
    #for epoch in range(3):
            optimizery.zero_grad()  # 기울기 초기화
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()  # 역전파
            optimizery.step()  # 역전파 단계에서 수집된 변화도로 매개변수를 조정

            #print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    origin = []
    corr = 0

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack / 이미지의 기울기 값을 구하도록 설정
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad / 이미지의 기울기값을 추출
        data_grad = data.grad.data

        # Call FGSM Attack / 교란된 이미지를 생성
        perturbed_data = fgsm_attack(data, epsilon, data_grad)


        # Re-classify the perturbed image/ 아마도 더 틀렸을 것. 입력이 일부러 down grad된 거니까
        train(model, perturbed_data, target)  #교란된 이미지를 학습시킴

        output = model(perturbed_data)
        out = notTrained(perturbed_data)
        # Check for success
        final = out.max(1, keepdim=True)[1]
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print("notTrained: {}".format(final))
        #print("Trained: {}".format(final_pred))
        if final.item() == target.item(): #정답일때
            corr += 1
            if (epsilon == 0) and (len(origin) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy() #교란된 이미지의 행렬?
                origin.append((final.item(),adv_ex))
        else: #정답이 아닐때
            # Save some adv examples for visualization later
            if len(origin) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                origin.append((final.item(), adv_ex))
        if final_pred.item() == target.item():  #정답일때
            correct += 1
            # Special case for saving 0 epsilon examples / 이때는 epslion = 0이므로 원본 data가 들어간 것이나 다름 없음
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy() #교란된 이미지의 행렬?
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )  #초기 예측값, FGSM의 결과 예측값,교란된 이미지를 append
        else: #정답이 아닐때
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )  #초기 예측값, FGSM의 결과 예측값,교란된 이미지를 append


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))   #10000개 중에 몇 개 맞았느냐  Trained
    finalAcc = corr / float(len(test_loader))  # 10000개 중에 몇 개 맞았느냐  not Trained
    print("no Trained / Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, corr, len(test_loader), finalAcc))
    print("Trained / Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, finalAcc, adv_examples, origin

accuracies = []
examples = []
faccuracies = []
fexamples = []

# Run test for each epsilon
for eps in epsilons:
    acc, facc, ex , ori= test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
    faccuracies.append(facc)
    fexamples.append(ori)

plt.figure(figsize=(5,5))
#plt.subplot(1,2,1)
plt.plot(epsilons, accuracies, color='blue')  #Trained
plt.plot(epsilons, faccuracies, color='red')  #not Trained
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
#
# plt.figure(figsize=(5,5))
# plt.subplot(1,2,2)
# plt.plot(epsilons, faccuracies, "g")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, .35, step=0.05))
# plt.title("not Trained Accuracy vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
