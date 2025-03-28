# Simple neuron networks for Java
Shortened name SNN4J.

## What is this library?
SNN4J is simple library for Java, which allows you to create simple neuron networks, train them and test them.
Currently, it is still developed and the whole system is being reworked to allow for much more customizable networks.

## How do I download it?
1. If you use Maven, add to your pom.xml: 
```xml
<repository>
      <id>jitpack.io</id>
      <url>https://jitpack.io</url>
</repository>

<dependency>
      <groupId>com.github.Codin-bee</groupId>
      <artifactId>SimpleNeuronNetworks4J</artifactId>
      <version>LATEST-RELEASE</version>
</dependency>

```
2. If you use Gradle add to your build.gradle:
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

After that reload your build system files changes and the library should be in your folder with external libraries.

## How do I use it?

THIS IS TUTORIAL FOR OLD VERSION OF THE LIBRARY THE CURRENT SYSTEM WORKS USING LAYERS AND I WILL PROVIDE TUTORIAL ONCE IT IS FINISHED

My goal is to create as good documentation in the code as possible, so it is really easy to use. But I know 
how hard it can be to make the first step, so I will provide simple tutorial for the basic features.

First thing u have to do is create the network. We will use the MLP class for it, which stands for multi-layer perceptron which is one of the neural network architectures.
```Java
MLP myNetwork = new MLP(784, 10, new int[]{10, 10}, "my network");
```
The first parameter in the constructor defines how many neurons the input layer has, the second one defines how many neurons are in output layer. The array determines amounts of neurons in hidden layers. And the String is the name of the network in file system.

Next we have to generate random weights and biases, which we will tune later by training, but they have to be generated randomly at the start. We do it using this method:
```Java
myNetwork.initializeWithRandomValues("src/main/resources/networks");
```
At the given path the library creates directory with the name of the network specified in the constructor. Inside directory for every layer will be created, containing one text file for each neuron.

When we have the network in our file system we just load it into the actual network, simply by calling:
```Java
network.initializeFromFiles("src/main/resources");
```
Now our network is fully set-up, and we can do whatever we want with it. I will divide the functionalities into three sections.

### 1. Training, data preparation
First we have to prepare the dataset we will train our network on.
```Java
double[][] inputData = new double[1][];
double[][] expectedResults = new double[1][];
//Your own value initialization
Dataset data = new Dataset(inputData, expectedResults);
```
The first array are the values passed to the network and the second one are values we want to get after it process the input. Now we can train our network calling the train method.
```Java
myNetwork.train(data, 100, true);
```
First parameter is obviously the dataset, second one is number of iterations(epochs) and last one is boolean, deciding if you want to print debug info while training.

### 2. Processing - actual usage
The MLP can process your input two ways and return either index of the output neuron with the highest activation:
```Java
int index = myNetwork.processAsIndex(new int[]{1, 2, 3});
```
or it can return activations of all output neurons:
```Java
double[] activations = myNetwork.processAsValues(new double[]{1, 2, 3});
```

### 3. Debugging, analysing, tuning
I provided few methods, to help you analyze and debug your network.
1. You can get the cost function of your model simply, by calling:
```Java
double[][] inputData;
double[][] expectedResults;
//Data initialization
double cost = myNetwork.calculateAverageCost(inputData, expectedResults);
```
2. You can also get the correctness percentage of your model on given dataset by calling:

```Java
Dataset dataset = new Dataset();
//Data initialization
double cost = myNetwork.getCorrectPercentage(dataset);
```

### What about the other architectures?
All the object have the same interface methods, just in some cases they use different data-types, or the arguments are slightly changed up. The Javadoc documentation provided should be enough, but you can reach me to ask any questions if needed, do not be afraid.

## Have you got already working code I can use?
Of course here is link to my repository I used to train one of my projects.
I will link all the repositories as I test the individual aspects of the library myself.
https://github.com/Codin-bee/DigitRecognition

## I found a bug!
That can definitely happen, this library is still in development, and it will probably be for a long time.
Please feel free to contact me via GitHub or my e-mail thecodingbee.dev@gmail.com and provide some information about the bug, so I can fix it for you.


## Is this project abandoned ?
Not really, but I am a busy man with busy life and I do not have much time left for this project right now.
