# Simple neuron networks for Java
Shortened name SNN4J.

## What is this library?
SSN4J is simple library for Java, which allows you to create simple neuron networks, train them and test them. It currently supports the MLP(multi-layer perceptron) architecture of neural network. Basic example of project you could do with this library is for instance number recognition AI using the mnist database.

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

Then reload your build system files changes and the library should be in your external libraries folder.

## How do I use it?
My goal is to create as good documentation in the code as possible so it is really easy to use. But i know how hard it can be to make the first step so i will provide simple tutorial for the basic features.

First thing u have to do is create the network. We will use the MLP class for it, which stands for multi-layer perceptron whichis one of the neural network architectures.
```Java
MLP myNetwork = new MLP(784, 10, new int[]{10, 10}, "my network");
```
The first paramter in the constructor defines how many neurons the input layer has, the second one defines how many neurons are in output layer. The array determines amounts of neurons in hidden layers. And the String is the name of the network in file system.

## I found a bug!
That can definitely happen, this library is still in developement and it will probably be for a long time.
Please feel free to contact me via github or my e-mail thecodingbee.dev@gmail.com and provide some information about the bug so I can fix it for you.
