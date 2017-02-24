---
layout: post
title: A SIMPLE CNN EXPERIMENT
published: True
date: 2017-03-24
---


A simple convolutional neural network from scratch with Numpy as its only dependency.So let's get straight to it.The following are all the functions that are used to make this CNN.

### CONVOLVE:

![image of convolution](/convolution_schematic.gif)
 


Its the most important part of the CNN.So in this step we have a filter that strides along the image and produces an activation map.So the function to do this will take two inputs that is a 3d array say image and a 4d array say filter which contains many filter(3d array) and performs convolution with each filter and produces a activation map which is again a 3d array having depth as the number of filters.The code looks like this.


```python
def convolve(inp,filter,bias,stride=1):
	dep,h,w=inp.shape
	filnum,depth,fh,fw=filter.shape		
	od=(h-fh)/stride+1
	tot=int(od*od)
	output=np.zeros((filnum,tot))
```

I've used sigmoid as my activation function just for simplicity with backpropagation.So it takes an input input,filter and a bias.Stride length is one by default.So dimension of the output 'od' is calculated using the formula

```python
output dimension=(image size - filter size)/stride length+1
```
We then create the activation function 'output' whose depth is number of filters and then we stretch each filter into corresponding rows.Then comes the convolution part


```python

for d in range(filnum): 
		row=0
		slide=0
		for i in range(tot):
			inpart=inp[:,row:fw+row,slide:fw+slide]
			
			output[d][i]=np.sum(inpart*filter[d])+bias[d]
			
			slide+=stride
			if slide==od:
				slide = 0
				row+=stride
				if row==od:
					break
```

The about code strides the filter and creates the stretched out version of the activation map.The number of bias units is equal to the number of filters i.e, it has a dimension of (numberOfFilters,1).Then the output is reshaped into its proper shape and is returned after applying the nonlinearity.


```python
output=output.reshape((filnum,int(od),int(od)))
return sigmoid(output)
```

The full function looks like this.

```python
def convolve(inp,filter,bias,stride=1):
	dep,h,w=inp.shape
	filnum,depth,fh,fw=filter.shape
	od=(h-fh)/stride+1
	tot=int(od*od)
	output=np.zeros((filnum,tot))
	#print("bais shape:"+str(bias.shape))
	
	for d in range(filnum): 
		row=0
		slide=0
		for i in range(tot):
			inpart=inp[:,row:fw+row,slide:fw+slide]
			
			output[d][i]=np.sum(inpart*filter[d])+bias[d]
			
			slide+=stride
			if slide==od:
				slide = 0
				row+=stride
				if row==od:
					break

	output=output.reshape((filnum,int(od),int(od)))
	return sigmoid(output)

```


### MAXPOOL:

This is a great step which reduces the dimensionality of your data into half i.e, you do this by taking maximum of some square of fixed size throughout the activation map which looks something like this

image 


```python
def maxpool(inp):
	depth,height,width=inp.shape
	poolsize=int(height/2)*int(width/2)
	output=np.zeros((depth,poolsize))
	maxindex=np.zeros((depth,poolsize,2))
	
```

Function maxpool takes 'inp' a 3d array and produces two outputs i.e, the desired output after maxpooling and another array called 'maxindex' which contains the index of all the maximum values in the output which is very helpful while performing backpropagation.It will have same size as 'output' but in order to Even here we stretch each layer in the output into rows. The output will have the same depth as the input but one fourth the values.The key part of maxpooling looks like this

```python
for d in range(depth):
		row=0
		slide=0
		for i in range(poolsize):
			sam=inp[d][row:row+2,slide:slide+2]
			output[d][i]=np.amax(sam)
			index=[ind for ind in zip(*np.where(sam==np.max(sam)))]
			if len(index)>1:
				index=[index[0]]
			maxindex[d][i]=index[0][0]+row,index[0][1]+slide
			
			slide+=2
			if slide>=(int(height)):
				slide = 0
				row+=2
```

The code is really similar to convolution in the striding part.After a part of the activation map 'sam' is selected the maximum value if found and then stored in the flattened out array 'output'.Then the line 


```python
index=[ind for ind in zip(*np.where(sam==np.max(sam)))]
```

gives the index of the maximum value with respect to the selected square but only the first value is taken.So to get the index w.r.t the whole array we add variable row and column to the index of the max value respectively.Then this is stored in the array 'maxindex'.Then this is again reshaped into it's proper size and returned.

```python
output=output.reshape((depth,int(height/2),int(width/2)))
maxindex=maxindex.reshape((depth,int(height/2),int(width/2),2))
return [output,maxindex]
```

The full function looks like this

```python
def maxpool(inp):
	poolsize=int(height/2)*int(width/2)
	output=np.zeros((depth,poolsize))
	maxindex=np.zeros((depth,poolsize,2))
	for d in range(depth):
		row=0
		slide=0
		for i in range(poolsize):
			sam=inp[d][row:row+2,slide:slide+2]
			"""print("sam")
			print(sam)"""
			output[d][i]=np.amax(sam)
			index=[ind for ind in zip(*np.where(sam==np.max(sam)))]
			if len(index)>1:
				index=[index[0]]
			maxindex[d][i]=index[0][0]+row,index[0][1]+slide
			
			slide+=2
			if slide>=(int(height)):
				slide = 0
				row+=2			
	output=output.reshape((depth,int(height/2),int(width/2)))
	maxindex=maxindex.reshape((depth,int(height/2),int(width/2),2))
	return [output,maxindex]

```


### DELPOOLTOCONV:


This function takes deltas from the pooling layer and passes the deltas to the convolutional layers.The output is double the size but has the same depth.This is a straight forward step since the delta is passed to that neuron which had the maximum value.We know the index of the maximum value and hence can do this very effectively.It looks something like this

```python
def delpooltoconv(pooldelta,maxindex):
	depth,height,width=pooldelta.shape
	pooldelta=pooldelta.reshape((depth,height*width))
	convdeltas=np.zeros((depth,height*2,width*2))
	maxindex=maxindex.reshape((depth,height*width,2))
	for d in range(depth):
		for i in range(height*width):
			x,y=map(int,maxindex[d][i])
			convdeltas[d][x,y]=pooldelta[d][i]
	return convdeltas
```


### GRADOFFILTER:

The process of finding gradients of weights is a bit different in CNNs.Since we have weight sharing we will have more than one updates.So in order to find the final weight update we simply sum up all the updates associated with that weight.

```python
def gradoffilter2(delta,prevact):
	depth,height,width=prevact.shape
	noofdel,dh,dw=delta.shape
	od=(height-dh) +1
	gradient=np.zeros((noofdel,depth,od*od))
	gradbias=np.zeros((noofdel,1))
	for d in range(noofdel):
		row=0
		slide=0
		for i in range(od*od):
			inp=(prevact[:,row:row+dh,slide:slide+dw])
			xx=inp*delta[d]
			#print("xx shape:"+str(xx.shape))
			gradient[d][:,i]=np.array([np.sum(x) for x in xx])
			slide+=1
			if slide==od:
				slide=0
				row+=1
		gradbias[d]=np.sum(delta[d])

	return gradbias,gradient.reshape((noofdel,depth,od,od))
```

So in the function we form a numpy array 'gradient' of the filters shape and 'gradbias' with the shape of the bais and the n stretch it.Then in the for loops we loop through each 2d array in 'gradient' array.It is very similar to normal weight update but we just sum all the updates associated with that weight.


### ARCHITECTURE OF CNN:

image of architecture


Here the image is convolved by five filters of size 3X3 to form an activation map of 26x26 having a depth of 5 is size of (5,3,3).Then comes the max pooling layer which reduces the size of the activation map to (6,13,13) as the stride length is 2.Hence the size is reduced by 4 times.Then this maxpooled layer is stretched into a single column and is then fed as input to the fully connected layers.The fully connected layers consist of an input layer,one hidden layer and an output layer.The input layer has the size of (1,13*13*5),hidden layer is (1,25) and the output is (1,10) i.e, the number of classes.The weights between hidden layer and input layer 'w0' has a size of (13*13*5,25) and the weights between the hidden layer and the output layer has a size of (25,10).The filters and weights are initialized so as to have zero mean.

```python
fils1=2*np.random.random((fils1no,1,3,3))-1
w0=2*np.random.random((13*13*fils1no,hiddenno))-1
w1=2*np.random.random((hiddenno,10))-1
bias1=2*np.random.random((fils1no,1))-1
b0=2*np.random.random((1,hiddenno))-1
b1=2*np.random.random((1,10))-1
```
The input to this CNN is an image and output is a matrix of size (1,10),the maximum of which is the class to which the image belongs.
```python
			image=shapeprop(data[1],"i")/255
			label=labels[data[0]]
			act1=convolve(image,fils1,bias1)
			act2,ind2=maxpool(act1)
			h,d,w=act2.shape
			act5=act2.reshape((1,h*d*w))
			
			#act5=image.reshape((1,784))
			act6=sigmoid(np.dot(act5,w0)+b0)
			act7=sigmoid(np.dot(act6,w1)+b1)
			
			error=act7-label
```

### BACKPROPAGATION:

Similar to vanilla backpropagation we first find all the deltas of all the layers and then to get the weights updates by multiplying the layer's delta with the previous layer's activation.We then use vanilla gradient decent to optimise the loss.The loss is just the different between the predicted value and the label matrix

```python
			del7=error*sigmoidprime(act7)
			del6=del7.dot(w1.T)*sigmoidprime(act6)
			del5=del6.dot(w0.T)*sigmoidprime(act5)
			del2=del5.reshape(act2.shape)
			del1=delpooltoconv(del2,ind2)
			
			bias1grad,fil1grad=gradoffilter2(del1,image)
			w0grad=act5.T.dot(del6)
			w1grad=act6.T.dot(del7)
			b0grad=del6
			b1grad=del7
			
			gfils1-=fil1grad
			gbias1-=bias1grad
			gw0-=w0grad
			gw1-=w1grad
			gb0-=b0grad
			gb1-=b1grad


			batchno+=1
			if batchno%1==0:
				fils1+=lr*gfils1
				w0+=lr*gw0
				w1+=lr*gw1
				bias1+=lr*gbias1
				b0+=lr*gb0
				b1+=lr*gb1

				gfils1=np.zeros(fils1.shape)
				gw0=np.zeros(w0.shape)
				gw1=np.zeros(w1.shape)
				gbias1=np.zeros(bias1.shape)
				gb0=np.zeros(b0.shape)
				gb1=np.zeros(b1.shape)
```

The later half is only useful when doing mini-batch training.You may notice that del1 is not multiplied with derivative of that layer because I found it to work better that way.

[The full code can be found here]()

### LEARNT FEATURES:
After 37 epoches the filters in 'fils1' look something like this

images of filters 

### CONCLUSION AND RESULTS:

The CNN was trained on the 1st 1000 images of the MNIST dataset and tested on the 1st 100 images and after 37 epoches it could successfully classify 91 images correctly.A fully connected neural network was also trained on the same mini dataset for more than 100 epochs but it was able to classify only 86 images correctly.Just by adding three 3x3 filters there was a boost from 86 to 91 percent accuracy.But the downfall of the above ' toy cnn ' is that it is terribly slow to train.But the above experiment if you can call it one shows that CNNs through there filters really learn the essential features in a image.















