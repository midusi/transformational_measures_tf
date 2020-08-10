# Metrics on Neural Network Models in TF 2

In this library, there are a set of python classes, developed in tensor flow 2,designed to be used in the calculus of different metrics on neural network models.  
Besides, there are three metrics developed. These metrics are known as Invariance, FeaturesVariance, and Same Equivariance.

### Main Classes

1. [Model](#Class-Model)
2. [DataSet](#Class-DataSet)
3. [TransformationsDataSet](#Class-TransformationsDataSet)
4. [Iterator](#Class-Iterator)

### Others Classes

1. [ModelStructure](#Class-ModelStructure)
2. [LayerStructure](#Class-LayerStructure)
3. [Block](#Class-Block)
4. [MnistTransformationsDataSet](#Class-MnistTransformationsDataSet)
5. [Analyzer](#Class-Analyzer)
6. [Graphs](#Class-Graphs)

### Metrics Classes

1. [Variance](#Class-Variance)
2. [NormalizedVariance](#Class-NormalizedVariance)
3. [FeaturesVariance](#Class-FeaturesVariance)
4. [NormalizedFeaturesVariance](#Class-NormalizedFeaturesVariance)
5. [SameEquivariance](#Class-SameEquivariance)
6. [NormalizedSameEquivariance](#Class-NormalizedSameEquivariance)

<br/>
<br/>

## Class Model

### Introduction

Let's supose we have a neural network model, as in the figure 1:

![](docs/images/DiagramaModel1.png)

In detail, a neural network model is conformed by a series of layers as we can see in the figure 2:

![](docs/images/DiagramaModel2.png)

where every layer has a set of activations that we call $aij$ being i the layer index and j the activation index.

Now, let's suposse we are interested in knowing the value of each activation when our model predict an input X. To get these values it is necessary a model that has as outputs not only the output Y but also the activations $aij$.
One of the main features of the class Model is what we have mentioned before, that is to say, our class Model allows us getting the activations values corresponding to the prediction of any input X. This class takes a Keras Model and creates a new model having as outputs the activations for each layer in the Keras Model.
<br/>
<br/>
Another feature of the Class Model is that its method predict takes as argument a matrix of inputs. It was thougth to deal with metrics that works with samples and transformations, but it is also possible to forget the input meaning and just think of a 2D input.

### Usage

```python

model_keras = ...  #instance of a Keras Model
model = Model(model_keras=model_keras)
input = ... # tensor of rank n>2
output = model.predict(input)
i=...
j=...
l=...
print("Activations values for input[i][j] in the layer l " + output[(i,j,l)])
#Another way of instantiating the class is the next
path = ".../mymodel.h5" #path to the file .h5
model = Model(path=path)

```

### Methods

- predict
- get_model_structure
- get_name_layer
- evaluate

<br/>

#### Method: predict

**Arguments**

- tensor_input: a tensor of rank n, n > 2, where the first two dimentions represent the matrix and the rest dimentions represent the input data.
  <br/>

**Returns**

Returns an instance of Block. This instance can be indexed for the tuple (i,j,l) where i represents the row, j represents the column corresponding to the input matrix and l represents the l-layer of a model. The instance can be indexed too for the tuple (i,j,k,f) where in this case f represents the feature. 
See the description of the Class Block for a better understanding. 


#### Method: get_model_structure

Returns an instance of ModelStructure with the model structure.

#### Method: get_name_layer

**Arguments**
- k:int,the layer index
  
**Returns**
Returns the name of hte layer k.

#### Method: evaluate

**Arguments**
- x:tf.tensor, a tensor with model inputs
- y:tf.tensor, a tensor with the desired outputs

**Returns**  
Returns a list with the loss and the accuracy for the tuple(x,y)

---------


## Class DataSet

DataSet is an abstract class designed to represent a 2D dataset which will be used by the Iterator to iterate over the model. The implementations of the DataSet are responsability of the user and they are necessary to use the Iterator. An implementation available in this library is the MnistDataSet.

### Methods

- get_data_shape
- get_width
- get_height
- get_matrix
- transpose

#### Method: get_data_shape

Returns the shape of the data in the matrix.

#### Method: get_width

Returns the matrix width.

#### Method: get_height

Returns the matrix height.

#### Method: get_matrix

**Arguments**

- rows: a list of indices
- columns: a list of indices
  <br/>

**Returns**

Returns a submatrix conformed by the rows and columns passed as arguments.

#### Method: transpose

Transpose the DataSet.

#### Method: is_transpose

Returns if the dataset is transposed or not.




---------


## Class TransformationsDataSet

This class extends the abstraction of the Class DataSet in order to have a dateset oriented to transformations. It is an abstract class too, that inherits from the class DataSet. 
This class was thougth to have groups of transformations. These groups are setting up in the dataset through the set_transformation_group method. 

### Methods


#### Method: apply_inverse_transformation

It applies an inverse transform to a tensor

**Arguments**

- tensor: tf.Tensor. The tensor to apply the inverse transform
- index_transformation:int. The index of the transformation to apply the inverse


#### Method: get_size_transformations_groups

Returns the amount of transformations groups. 


#### Method: set_transformation_group

Sets the transformations group in the dataset

**Arguments**

- group: int. The group
- n:int. The amount of transformations for this group.




---------




## Class Iterator

### Introduction

Let's consider we have a 2D dataset, for example a dataset of samples and transformations, and we want to get every value predicted by the model for each input in the dataset. If we think of all possible values that we will obtain and besides we divide these values into layers, it is possible to imagine some kind of data representation in the way showing in the next figure:

![](docs/images/DiagramaIterador1.png)

Every square represents a matrix with the values of an activation for each input in the dataset. For instance, the first square represents the activation $a11$ and has the values $a11(ti(xj))$ with $i\ \varepsilon \ \{1,...,m\}$ and $j\ \varepsilon \ \{1,...,n\}$ where $a11(ti(xj))$ is the value of the activation $a11$ predicted by the model for the input $ti(xj)$.

<br/>
<br/>

Now that we have a visual representation of the values we are looking for, it is easier to understand what the Iterator Class does. The Iterator gives you the possibility of moving and getting the values through blocks of width and height defined by yourself.  
For example, let's consider a block of 3x3. The iterator allows you obtaining the data in the following way:

![](docs/images/DiagramaIterador2.png)
![](docs/images/DiagramaIterador3.png)
![](docs/images/DiagramaIterador4.png)

### Usage

```python
model = Model(...)
dataset = DataSet(...)
iterator = Iterator(model, dataset)
height = 3
width = 3
for height, blocks in iterator.get_block(height, width):
    #...
    for width, block in blocks
        #...    
```





#### Look at the Variance Class to have another using example!

### Constructor

**Arguments**

- model: an instance of the Model class
- dataset: an instance of the DataSet class

<br/>

### Methods

- get_block
- get_model


#### Method: get_block

**Arguments**

- height: the height of the block
- width: the width of the block
<br/>

**Returns**

Returns a "vertical" iterator where each item in this iterator is a tuple (height,blocks_iterator) where blocks_iterator is an "horizontal" iterator in which each item is a tuple (width,block_activations) where block_activations is an object that indexes a three dimensional list through a tuple (i,j,l) where i represents the row and j represents the column corresponding to the block and l represents the l-layer of a model. The values in the list are the activations values in the layer l for the input located in the row i and the column j, and width is the width of the block and the height of the block is the height returned previously.  

***Remark***

When we say "vertical" iterator, we refer to the movement of the iterator in the block, the same happens with we say "horizontal" iterator.  


#### Method: get_model

Returns the model in the iterator. 


---------


## Class ModelStructure

### Introduction

The goal of this class is to represent the structure of a keras model. On this way, the class ModelStructure has the structure of each layer in the model and it has too a map between layers and layers with features. The last information allows us to define the features variance metric. 


### Constructor

**Arguments**
- model: an instance of a keras model

<br/>

### Attributes

- layers:list. A list with an instance of LayerStructure for each layer
- number_layers:int. The number of layers in the model
- map_layers_features:list. A list containing a map between the layers and the layers with features
- layers_features:list. A list with an instance of LayerStructure for each layer with features
- number_layers_features:int. The number of layers with features


### Methods (No one)


---------

## Class LayerStructure

### Introduction

This class contains the information of the layer structure.

### Attributes

- number_features:int. The number of features in the layer
- shape_feature:tuple. The shape for the features in the layer
- number_activations:int. The number of activations in the layer
- shape:tuple. The shape of the layer

### Methods (No one)



---------

## Class Block

### Introduction
   
   This class is used by the model to encapsulate the output in a block when it is predicted a block of inputs.The block can be indexed by a tuple of three or four indexes to get a particular output.  
   The tuple (i,j,k) gives the output of the layer k for the input in the block given by the row i and the column j.  
   The tuple (i,j,k,f) gives the output of the feature f in the layer k for the input in the block given by the row i and the column j.  
   Besides, the block has some information of the dataset from which the inputs come. 
  

### Attributes

- height:int. The height of the block
- width:int. The width of the block
- model_structure:ModelStructure. The model structure


### Methods
 
#### Method: set_rows_dataset
This method can be used to set the corresponding dataset rows in the block.

**Arguments**
- rows: list

#### Method: set_columns_dataset
This method can be used to set the corresponding dataset columns in the block.

**Arguments**
- columns: list

#### Method: set_state_transpose_dataset
This method can be used to set if the dataset is transposed.

**Arguments**
- state: bool

#### Method: get_rows_dataset
Returns the corresponding dataset rows in the block.

#### Method: get_columns_dataset
Returns the corresponding dataset columns in the block.

#### Method: is_dataset_transpose
Returns the transpose state of the dataset.


<br/>



---------

## Class MnistTransformationsDataSet

This class implements the abstract class TransformationsDataSet. This dataset is Mnist with transformations. It has 8 groups of transformations. These groups are:  

1. Identity
2. Rotations
3. X-Translations
4. Y-Translations
5. X,Y-Translations
6. Rotations and X-Translations at the same time
7. Rotations and Y-Translations at the same time
   

---------

## Class Analyzer

### Introduction

This class tries to analyze if there are relations between the results of a metric and the accuracy. For example, if you take the variance metric and a group of transformations for a dataset, you can see if when the variance increases the accuracy decreases. 

### Constructor

**Arguments**

- model:Model. 
- dataset:TransformationsDataSet. 
- iterator:Iterator. 

### Methods
 
#### Method: run
This method run the analyses.

**Arguments**

- n:int. The amount of transformations for each group in the dataset. 


### Usage

```python
model = Model(...)
dataset = TransformationsDataSet(...)
iterator = Iterator(model,dataset)
analyzer = Analyzer(model,dataset,iterator)
analyzer.run(100)
```



---------

## Class Graphs

---------

## Class Variance


### Introduction

Let's A be an activation in our model. One way to define the variance of A for a DataSet X is as follows:

Var(A,X)="The mean of the variance for every row in X", i.e., 

![](docs/images/variance1.png)

Similarly, it can be defined the variance for a Layer in the Model. 

![](docs/images/variance2.png)

So, the class variance computes these values for a dataset X and a model M using the Iterator. 

### Usage

```python
model = Model(...)
dataset = DataSet(...)
iterator = Iterator(model, dataset)
variance = Variance(Iterator)
variance.compute()
print(variance.variance_layers)
```

### Constructor

**Arguments**

- iterator: it is an instance of the Iterator class

<br/>

### Methods

- compute


#### Method: compute

**Arguments**

- height: it is the height of the block to iterate
- width: it is the width of the block to iterate
<br/>

**Returns**

Returns a tuple (variance_layers,variance_layers_activations) where:

- variance_layers: it is a list with the variance of every layer
- variance_layers_activations: it is a list where each element in the list is a list with the variance of every activation in the respective layer 

---------

## Class NormalizedVariance


---------

## Class FeaturesVariance

---------

## Class NormalizedFeaturesVariance

---------


## Class NormalizedSameEquivariance


---------

## Class NormalizedFeaturesVariance
