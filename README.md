# Metrics on Neural Network Models in TF 2

In this library, there are a set of python classes, developed in tensor flow 2,designed to be used in the calculus of different metrics on neural network models.  
Besides, there are two metrics developed. These metrics are known as Invariance and Same Equivariance.

### Main Classes

1. Model
2. DataSet
3. Iterator

### Metrics Classes

1. Variance
2. Same Equivariance

<br/>
<br/>

## Class Model

### Introduction

Let's supose we have a neural network model, as in the figure 1:

![](DiagramaModel1.png)

In detail, a neural network model is conformed by a series of layers as we can see in the figure 2:

![](DiagramaModel2.png)

where every layer has a set of activations that we call $aij$ being i the layer index and j the activation index.

Now, let's suposse we are interested in knowing the value of each activation when our model predict an input X. To get these values it is necessary a model that has as outputs not only the output Y but also the activations $aij$.
One of the main features of the class Model is what we have mentioned before, that is to say, our class Model allows us getting the activations values corresponding to the prediction of any input X. This class takes a Keras Model and creates a new model having as outputs the activations for each layer in the Keras Model.
<br/>
<br/>
Another feature of the Class Model is that its method predict takes as argument a matrix of inputs. It was thougth to deal with metrics that works with samples and transformations, but it is also possible to forget the input meaning and just think of a 2D input.

### Usage

<code>
model_keras = ...  #instance of a Keras Model
<br/>
model = Model(model_keras=model_keras)
<br/>
input = ... # tensor of rank n>2
<br/>
output = model.predict(input)
<br/>
i=...
<br/>
j=...
<br/>
l=...
<br/>
print("Activations values for input[i][j] in the layer l " + output[(i,j,l)])
<br/>
<br/>
#Another way of instantiating the class is the next
<br/>
path = ".../mymodel.h5" #path to the file .h5
<br/>
model = Model(path=path)

</code>

### Methods

- predict
- size_layers
- size_activations
- size_activations_layers

<br/>

#### Method: predict

**Arguments**

- tensor_input: a tensor of rank n, n > 2, where the first two dimentions represent the matrix and the rest dimentions represent the input data.
  <br/>

**Returns**

Returns an object that indexes a three dimensional list through a tuple (i,j,l) where i represents the row and j represents the column corresponding to the input matrix and l represents the l-layer of a model. The values in the list are the activations values in the layer l for the input located in the row i and the column j.

#### Method: size_layers

Returns the amount of layers in the model.

#### Method: size_activations

Returns the amount of activations in the model.

#### Method: size_activations_layers

Returns a list with the amount of activations in each layer of the model.
