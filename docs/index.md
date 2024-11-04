# Welcome to IREE.gd Documentation

Here resides all the important stuff to use [IREE.gd](https://github.com/iree-gd/iree.gd), the GDExtension for running machine learning model natively in Godot.
IREE.gd is made from [IREE](https://iree.dev), the machine learning model compiler and runtime suite, and [Godot-cpp](https://github.com/godotengine/godot-cpp), aka. GDExtension for C++.

IREE.gd requires Godot 4.2 to work. For the supported platforms, IREE.gd supports multiple major desktop and edge platforms. 
It uses different backends to accelerate the computation depending on the platform for running the machine learning model.
Backends are executors for running the model. The backends are Vulkan, Metal and VMVX.
Below are the supported platforms with its corresponding backend.

| Platform | Backend |
| ----- | ----- |
| MacOS, iOS | Metal |
| Windows, Linux, \*BSDs, Android | Vulkan |

As a side note, for the Metal backend, IREE.gd only supports apple platform with apple silicon.

For any inquiry, you can enter the [Godot Scientific Discord Server](https://discord.gg/zgSjGPDNKP) for a chat. Any feature request or bug report could be done in the [Github issue page](https://github.com/iree-gd/iree.gd/issues).

