#! /usr/bin/env python
from flask import Flask, render_template, request, Response
import numpy as np
from binascii import a2b_base64
import imageio
from PIL import Image
import io
import time
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

global model_states, nb_epoch #to have access later
model_states = ['Not Trained']
nb_epoch=5


app = Flask(__name__)

model =None
#page to_train
@app.route('/')
def to_train():
    return render_template('to_train.html', nb_epoch=nb_epoch)

#train the model
@app.route("/loadmodel/", methods=['GET'])
def load():
	global model 
	class NN(nn.Module):
	  def __init__(self):
	    super(NN, self).__init__()
	    #First parameter is no. of input channels. For MNIST, input is 1 channel. Output no. of channel can be anything. Let's take 20. We'll get 26x26x20 from input of 28x28x1 of MNIST
	    #The third parameter is kernel size
	    #Fourth parameter is stride. Here, taking 1 stride
	    self.conv1L= nn.Conv2d(1, 20, 3, 1) 

	    #Second layer
	    self.conc2L= nn.Conv2d(20,50,3, 1) #Input channel=20 of previous output, output=50 channels, usually we use 3x3 kernel and stride is 1
	    
	    #Let's take Third layer as fully connected
	    self.FC1=nn.Linear(5*5*50, 500)   #See video for what it means incl the network we are trying to bring up
	    
	    self.FC2=nn.Linear(500,10) #Taking the output from 500 to 10 for final layer for MNIST
	    
	    
	  def forward(self,x):
	    #Note that in pytorch the size tensor is (Channel x Size) not (Size x Channel)
	    x = F.relu(self.conv1L(x)) #Taking first layer and doing Relu activation------> Output Size: 20x26 = 20x26x26
	    x = F.max_pool2d(x,(2,2)) #Doing a max-pooling operation of 2x2 ------> Output Size: 20x13
	    x = F.relu(self.conc2L(x)) #Doing second convolution and Relu on the output------> Output Size: 50x11
	    x = F.max_pool2d(x,(2,2))#------> Output Size: 50x5 (maxpooling will take the floor value of Floats)
	    x = x.view(-1, 50*5*5) #Flattening the output for inputting to Linear layer. It makes the inout size to 1D array....view is used to flatten. -1 is used to automatically find out the row value. The column value should be input by us
	    x = self.FC1(x) #DoingFully connected convolution------> Output Size: 500 
	    x = self.FC2(x) #DoingFully connected convolution------> Output Size: 10 

	    return F.log_softmax(x, dim=1) #Doing Softmax operation on our 10 outputs and returning it

	checkpoint = torch.load("/Users/anandgokul/Desktop/Entirety/mnist_deployed_pytorch/Meetup_MNIST.pt")
	model = NN()
	model.load_state_dict(checkpoint)
	print("model loaded")
	return "Loading done"
#page where you draw the number
@app.route('/index/', methods=['GET','POST'])
def index():
    prediction='?'	
    if request.method == 'POST':

        dataURL = request.get_data()
        drawURL_clean = dataURL[22:]
        binary_data=a2b_base64(drawURL_clean)
        img = Image.open(io.BytesIO(binary_data))
        img.thumbnail((28,28))
        img.save("/Users/anandgokul/Desktop/Entirety/mnist_deployed_pytorch/data_img/draw.png")

    return render_template('index.html', prediction=prediction)

#display prediction
@app.route('/result/')
def result():
    time.sleep(0.2)
    img = Image.open("/Users/anandgokul/Desktop/Entirety/mnist_deployed_pytorch/data_img/draw.png").convert("1")
    transform=transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = torch.unsqueeze(img , 0)
    prediction = inference(model , img)
    print(prediction)
    return render_template("index.html",prediction=prediction)

def inference(model , img):
  output = model(img)
  output = torch.exp(output)
  top_prob,top_class=output.topk(1,dim=1)
  return top_class.item()

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
