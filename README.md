# Simple neuron networks for Java
Shortened name SNN4J.

## What is this library?
SNN4J is a library for Java, which allows you to create simple neuron networks,
train them, testem and use them for processing any data.
It is designed to give you control over as many aspects of the models as possible,
while having simple interface abstracting from all the complicated math.

## How do I download it?
Currently, the library uses Jitpack.io for the distribution.
In the future, it will probably be distributed on the Maven central.

### 1. If you use Maven, add to your pom.xml: 
```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>com.github.Codin-bee</groupId>
        <artifactId>SimpleNeuronNetworks4J</artifactId>
        <version>LATEST_RELEASE</version>
    </dependency>
</dependencies>
```
### 2. If you use Gradle, add to your build.gradle:
```groovy
dependencyResolutionManagement {
		repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
		repositories {
			mavenCentral()
			maven { url 'https://jitpack.io' }
		}
	}

dependencies {
		implementation 'com.github.Codin-bee:SimpleNeuronNetworks4J:LATEST_RELEASE'
	}

```

After pasting this, reload your build system files changes and the library should be in your folder with external libraries.

## How do I use it?

### 1. Layers
The architecture of all models accepts Layer implementations to use for the computations.
Every Layer has its own constructor and accepts different parameters.
Here are a few different examples:
#### Fully Connected Layer
Or also called dense layer, feed forward network or multi-layer perceptron.

```Java
FullyConnectedLayer ffn = new FullyConnectedLayer(784, 10, new int[]{16, 16}, 1, new ReLU());
```
In the example above, we create a layer with input dimension 784, output dimension 10,
two hidden layers of dimensions 16, with only one vector in the sequence and ReLU as activation function
(more about activation functions in the next chapter).

### 2. Activation Functions, Cost Functions and Weight Generators
There are different interfaces throughout the library making it more customizable.
You can implement them, however, you want if you are looking for certain function, the library does not provide.

#### a) Activation Functions
This function is part of every Layer it is a non-linear function applied to the results of the computed results. 
Some have special properties, and you can modify them using getters and setters.

#### b) Vector Activation Functions
They serve the exactly same purpose as normal Activation Functions, but they are made for vectors and therefore take into account all its elements to calculate the activation of others.

#### c) Cost Functions
This function is used to calculate the error of the network on given data, by comparing the predicted outputs and targeted outputs.

#### d) Weight Generators
To initialize the parameters of the Layers before training, we use Weight Generators.
They take the number of inputs and outputs of given layer as an argument to produce the best, numerically stable values.

### 3. Model
As a last step, we have to put our layers into a certain model.
Currently, the only option is the LayeredModel.
To use it, create a new instance with ArrayList of your layer as a constructor parameter:

```Java
LayeredModel model = new LayeredModel(List.of(myLayer1, myLayer2));
```
Optionally, you can call an empty constructor and later use the method addLayer:
```Java
model.addLayer(myLayer);
```
You can also specify the index of the layer:
```Java
model.addLayer(myLayer, index);
```
If you want, you can add an output activation function to the model.
It accepts any class implementing the VectorActivationFunction interface.
The most common is the softmax.

```Java
model.setOutputActivationFunction(new Softmax());
```

### 4. Training, Data Preparation
#### Data
Currently, the library cannot process the training data into usable values, but there will be solutions added later on.
If we want to make a dataset, we have to prepare our own data.
With it, we can create a Dataset object.
The data has two parts: inputs and prediction targets.
Both are stored in 3-dimensional arrays.
```Java
Dataset data = new Dataset(inputs, targets);
```
The arrays are indexed:
```Java
float[][][] inputs = new float[sample][vector-in-sequence][elements-of-vector];
```

#### Training
The train() method in the Model interface takes several parameters:
trainDataset - Dataset object instance, containing the training data
epochs - number of epochs/iterations for the training process
savePath - path for saving temporary progress
saveInterval - interval for saving the progress (number of epochs)
debugPrint - boolean value, decide whether to print out the cost, etc
```Java
model.train(trainDataset, epochs, savePath, saveInterval, debugPrint);
```

### 5. Processing—Actual Usage

### 6. Debugging, Analysing

## Have you got already working code I can use?
Of course, here is a link to my repository I used to train one of my projects.
I will link all the repositories as I test the individual aspects of the library myself.
https://github.com/Codin-bee/DigitRecognition

## I found a bug!
That can definitely happen, this library is still in development, and it will probably be for a long time.
Please feel free to contact me via GitHub or my e-mail thecodingbee.dev@gmail.com and provide some information about the bug, so I can fix it.