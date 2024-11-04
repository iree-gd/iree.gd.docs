# IREE.gd

Time to get started on spicing up your game with machine learning models!
Before we start, we will first set up IREE.gd in a project.
Then, we cover some concepts so it is easier to grasp what is going on when following the tutorial for people who are interested.
Lastly, we demonstrate how we use a machine learning model in Godot via IREE.gd. Unfortunately, there are several machine learning frameworks such as tensorflow and pytorch in the world of machine learning with some major differences among them. Thus, each framework would have its own dedicated way of using its model. In this documentation, we do cover models from these machine learning frameworks:

- [Tensorflow lite](#using-tensorflow-lite-model)

## Getting started
To get IREE.gd, go to [the official releases on Github](https://github.com/iree-gd/iree.gd/releases) and fetch a copy of IREE.gd. There are two versions:

- `iree-gd-sample-*.zip` - A Godot project with IREE.gd and several samples to test it out, good for trying out.
- `iree-gd-*.zip` - Lightweight version that only consist of the compiled libraries, good for integration into existing project.

In this documentation, we will be using `iree-gd-sample-*.zip` for quickly setup, and also able to test out whether IREE.gd is working properly or not.

### Testing out the samples

Before proceeding, it would be better to test out whether your IREE.gd is working on your device. 
After download and extract the aformentioned `iree-gd-sample-*.zip`,  open the project with Godot.
Hopefully, you'll be greeted with a baboon face without any errors.

![First time open sample](images/first_time_open_sample.png)

Just run the baboon scene. 

![Low resolution baboon](images/baboon_lowres.png)

Press the `upscale` button. If your baboon face becomes much clearer, congratuation, you have successfully run the [Enhanced Super Resolution GAN](https://www.kaggle.com/models/kaggle/esrgan-tf2) model!

![High resolution baboon](images/baboon_highres.png)

Later on, we will discuss on how the tensorflow lite model is imported into Godot.

## How IREE.gd really works

In this section, some technical details are discussed so you have a clearer image on what is going on when you import your own model, and hopefully in a simpler terms. Some of the details are abstracted away to make things simpler. For further reading, you can visit the [official IREE website](https://iree.dev). It is not mandatory and you could skip this section for now. 

IREE.gd, as the name suggested, is [IREE](https://github.com/iree-org/iree) ported into Godot. IREE itself is a compiler and runtime library specialized for running machine learning models. In a sense, it is like the Java compiler and Java virtual machine, the compiler produces bytecodes while the runtime runs the bytecodes. For the Java compiler, the language would be Java, while for IREE, the language is MLIR, a kind of language specialize for compiler to read instead of being user-friendly.
While for the runtime, instead of having something like `System.out.println` in Java for interacting with the user, the IREE runtime is specialized for interacting with the CPU or GPU, and make crazy fast accelerated linear algebra (which most of the machine learning model needs).

To use a machine learning model with IREE, one would need to turn, or transpile, the model into MLIR code that uses library (CPU or GPU functions) supported by the runtime. The library supported by the runtime depends on the application, for IREE.gd, there are Metal for apple products, Vulkan for Windows, Unix-like and android, and VMVX for the rest. So, later on, you'll need to take note on making your model target the platform that your game will be on (usually metal and vulkan will support most of the platforms). For turning model into MLIR code, there are different tools for different machine learning framework, as each framework has its own format of storing the  machine learning model, [too bad there is no standardized format for it](https://xkcd.com/927/). Thus, there are different way of porting the models for each machine learning framework.

After successfully generate the bytecodes, we will need to figure out the input format and output format. In IREE, the input data is in tensor, or multi-dimensional array. The input and output dimension would depends on the model. Later on, we will discuss how to inspect the input and output dimensions. Moreover, the user would need to figure out the meaning of the output, whether interpret as an image, or as a series of attributes. It would also heavily depends on the model. Unfortunately, there is no standard way of represent data, so one would need to figure it out, usually with the help of original example code. In this documentation, we will do our best to demonstrate the process of figuring out the attributes.

## Using Tensorflow lite model
In this section, we will be porting a tensorflow lite model called [Enhanced Super Resolution GAN](https://www.kaggle.com/models/kaggle/esrgan-tf2).



*Coming soon...*
