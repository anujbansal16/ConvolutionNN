#!/usr/bin/env python
# coding: utf-8

# # Assignment-6

# In[124]:


import pandas as pd
import numpy as np
from skimage.transform import resize
from scipy import misc
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ### -Load sample image

# In[119]:


sampleImg=mpimg.imread("linux-penguin.png")
print(sampleImg.shape)
plt.show(plt.imshow(sampleImg))


# ### -Activation functions

# In[88]:


def sigmoid( x):
  return np.divide(1, np.add(1, np.exp(np.negative(x))))
    
def relu(x):
  return (x) * (x > 0)
    
def tanh(x):
  return np.tanh(x)

def softmax(x):
  res = np.exp(x)
  return res/np.sum(res)


# ### -Weights initializer

# In[89]:



def initializeFilter(dim):
    stddev = 1/np.sqrt(np.prod(dim))
    return np.random.normal(loc = 0, scale = stddev, size = dim)
  
def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01


# ### Part-1 Forward pass of CNN
# 
# #### Approach:
# - Given an image of dimension (width,height,channels) we perform the **first convolution** with $6$ filters of size $5x5$ followed by an activation function to generate feature map `C1`.
# - Now the feature map `C1` is downgraded/subsampled using **maximum pooling** with window of dimension $2x2$ to generate subsamples `S2`.
# - On `S2` (subsampled image)  we perform the **second convolution** with $16$ filters of size $5x5$ followed by an activation function to generate feature map `C3`.
# - Now the feature map `C3` is downgraded/subsampled using **maximum pooling** with window of dimension $2x2$ to generate subsamples `S4`.
# - Now we apply **fully connected convolution layer** followed by an activation function on subsample `S4`. Here we use 120 filters to generate a flatten image.
# - Now on above flatten image we apply the **fully connected layer** of $84$ neurons which is then connected with an **output layer** of $10$ neurons (each represnting one class) operating with softmax function to generate class probabilities.

# In[110]:


def forward(image,filter1, filter2,weights3, weights4,weights5,b1,b2,b3,b4,b5,acti=relu):
    conv1 = convolution(image, filter1,b1)
    conv1=acti(conv1)    
    
    print("===========covolution 1==========",conv1.shape)
    fig=plt.figure(figsize=(16, 16))
    for i in range(conv1.shape[2]):
      fig.add_subplot(1,conv1.shape[2],i+1)
      plt.imshow(conv1[:,:,i])
    plt.show()
    
    
    
    downgrade1=maxPool(conv1)
    print("========== Max pool 1 =============",downgrade1.shape)
    fig=plt.figure(figsize=(16, 16))
    for i in range(downgrade1.shape[2]):
      fig.add_subplot(1,downgrade1.shape[2],i+1)
      plt.imshow(downgrade1[:,:,i])
    plt.show()
    
    
    conv2 = convolution(downgrade1, filter2,b2)
    conv2=acti(conv2)    
    print("===========covolution 2==========",conv2.shape)
    fig=plt.figure(figsize=(16, 16))
    for i in range(conv2.shape[2]):
      fig.add_subplot(1,conv2.shape[2],i+1)
      plt.imshow(conv2[:,:,i])
    plt.show()
      
    
    
    downgrade2=maxPool(conv2)
    print("========== Max pool 2 =============",downgrade2.shape)
    fig=plt.figure(figsize=(16, 16))
    for i in range(downgrade2.shape[2]):
      fig.add_subplot(1,downgrade2.shape[2],i+1)
      plt.imshow(downgrade2[:,:,i])
    plt.show()
    
    ######### convolution 3 fully connected
    
    conv3 = convolution(downgrade2, filter3,b3)
    conv3=acti(conv3)    
    print("===========covolution 3==========",conv3.shape)
    fig=plt.figure(figsize=(16, 16))
    flattenD = conv3.reshape((conv3.shape[2],1)) #flatten
    plt.show(plt.imshow(flattenD.T))
    
        
#     w,h,nf=downgrade2.shape
#     flattenD = downgrade2.reshape((w*w*nf,1)) #flatten
    
#     z1=weights3.dot(flattenD)+b3 #+ b3 # first Fully connected
#     z1=acti(z1) 
    
#     print("========== First Fully connected layer =============",z1.T.shape)
#     fig=plt.figure(figsize=(16, 16))
#     plt.show(plt.imshow(z1.T))
# #     print(z1)

    z2 = weights4.dot(flattenD)+b4 #+ b4 # second Fully connected
    z2=acti(z2) 
    
    print("========== Second Fully connected layer =============",z2.T.shape)
    fig=plt.figure(figsize=(16, 16))
    plt.show(plt.imshow(z2.T))
    
    finalOut = weights5.dot(z2)+b5 #+ b4 # third output layer
#     print(finalOut)
    probs=softmax(finalOut) # predict class probabilities
    print("=============Class Prediction Probabilities============",probs.shape)
    print(probs)
    
    
    


# ### Convolution Function

# In[111]:


def convolution(image, filterr, bias):
  m,n,channel=image.shape
  numofFilt,f,s,channels=filterr.shape
  outputDim=m-f+1
#   print(f)
  output=np.zeros((outputDim,outputDim,numofFilt))
 
  for count in range(numofFilt):
    for x in range(outputDim):
      for y in range(outputDim):
        output[x,y,count]=np.sum(filterr[count,:,:]*image[x:x+f,y:y+f,:])#+bias[count]
  return output    
    
  
  


# ### Maximum Pooling Function

# In[112]:


def maxPool(image,f=2,s=2):
#   print("Pool ",image.shape)
  m,n,channel=image.shape
  outputDim=int((m-f)/s)+1
  output=np.zeros((outputDim,outputDim,channel))
  for count in range(channel):
    for x in range(outputDim):
      for y in range(outputDim):
        output[x,y,count]=np.max(image[s*x:s*x+f,s*y:s*y+f,count])
  return output    


# ### -Parameter initialization

# In[120]:


numoffilt1=6 #number of filters in filter1
numoffilt2=16 #number of filters in filter2
channels=sampleImg.shape[2] #number of channel in image
f=5 # w=h of filter
wshape=int((((sampleImg.shape[0]-4)/2-4)/2))

# filter1, filter2, weights3, weights4,weights5 = (numoffilt1 ,f,f,channels), (numoffilt2 ,f,f,numoffilt1), (120,wshape), (84, 120),(10, 84)
filter1, filter2, filter3, weights4,weights5 = (numoffilt1 ,f,f,channels), (numoffilt2 ,f,f,numoffilt1), (120 ,wshape,wshape,16), (84, 120),(10, 84)

#filter and weight initialization (normal distribution)
filter1=initializeFilter(filter1)
filter2=initializeFilter(filter2)
filter3=initializeFilter(filter3)
weights4 = initializeWeight(weights4)
weights5 = initializeWeight(weights5)

#biases at each layer
b1=np.random.randn(*(filter1.shape[0],1))
b2=np.random.randn(*(filter2.shape[0],1))
b2=np.random.randn(*(filter3.shape[0],1))
b4=np.random.randn(*(weights4.shape[0],1))#*0.01
b5=np.random.randn(*(weights5.shape[0],1))#*0.01


# #### Relu as activation function

# In[121]:


forward(sampleImg,filter1,filter2,filter3,weights4,weights5,b1,b2,b3,b4,b5,relu)


# #### Sigmoid as activation function

# In[122]:


forward(sampleImg,filter1,filter2,filter3,weights4,weights5,b1,b2,b3,b4,b5,sigmoid)


# #### Tanh as activation function

# In[123]:


forward(sampleImg,filter1,filter2,filter3,weights4,weights5,b1,b2,b3,b4,b5,tanh)


# ### Part-2

# #### 1. What are the number of parameters in 1st convolutional layers ?
# **Ans**
# 
# Considering a colored image with 3 channels
# 
# There are $6$ filters in first convolution layer.
# 
# Dimension of each filter is $5x5x3$.
# 
# Each filter have $1$ bias term.
# 
# Therefore, parameters required for one filter is $(5*5*3)+1 = 76$.
# 
# Therefore, **number of parameters in 1st convolutional layers** $= 76*6 = 456$ **Ans**

# <hr>
# 
# #### 2. What are the number of parameters in pooling operation?
# **Ans** : No paramters are required , as pooling is just an operation of taking out maximum value in a window.

# <hr>
# 
# #### 3. Which of the following operations contain most number of parameters?
# 
# **Ans:**
# 
# Considering image of dimension $32x32x3$
# 
# Parameters in 1st convolution layer = $ 456 $
# 
# Parameters in 2nd convolution layer = $ 16*[(5*5*6)+1] = 2416  $
# 
# **Hghest Parameters in 3nd convolution layer (fully connected) = $ 120*[(5*5*16)+1] =   $ 48120**
# 
# Parameters in second fully connected layer = $ 84*[ 120 + 1] = 10164  $
# 
# Parameters in output = $ 10*[ 84 + 1] = 850  $
# 
# Activation Functions doesnt require any parameters.

# <hr>
# 
# #### 4. Which operation consume most amount of memory?
# **Ans:** 
# 
# Memory consumed mainly depends on 2 things:
# - Total Pixels
# - Total Parameters
# 
# As the number of parameters for the fully connected layer is highest thus, **Fully connected layers at the end consumes the highest memory.**

# <hr>
# 
# #### 5. Try different activation functions and describe observations.
# **Ans:** 
# 
# Outputs are shown in part-1 itself with reLU, sigmoid and tanh as activation functions. 
# 
# **Observations**
# 
# Application of differernt activation function at each of the layer doesnt affect the values of final probabilities as such.
# <br>
# Instead, the difference appears clearly on the feature map produced by the convolutions layers.
# **Ranges of different activation functions**
# 
# **ReLU**: $ [0,INF] $<br>
# **Sigmoid**: $ [0,1] $<br>
# **Tanh**: $ [-1,1] $<br>
# 
# As seen above, since tanh is just an scaled version of sigmoid , the output of convolution layer for each of this activation is nearly same.<br>
# But corresponding to the same weight values if we use ReLU, then it mask out the negative values (make them zero) thus the pixels corresponding to negative values are masked to `0` thus it shows dark regions corresponding to those values as show in outputs.
# 
# 
