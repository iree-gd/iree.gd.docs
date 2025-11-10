# Welcome to IREE.gd Documentation

Here is all the essential knowledge to use [IREE.gd](https://github.com/iree-gd/iree.gd), the GDExtension for natively running the machine learning model in Godot.
IREE.gd is made from [IREE](https://iree.dev), the machine learning model compiler and runtime suite, and [Godot-cpp](https://github.com/godotengine/godot-cpp), aka. GDExtension for C++.

IREE.gd requires Godot 4.2 to work.

For the supported platforms, IREE.gd supports multiple primary desktop and edge platforms. It uses different backends to accelerate the computation, depending on the platform on which the machine learning model is run.

Backends are executors for running the model. The backends are Vulkan, Metal and VMVX.

Below are the supported platforms with their corresponding backend.

| Platform | Backend |
| ----- | ----- |
| MacOS, iOS | Metal |
| Windows, Linux, \*BSDs, Android | Vulkan |
| The rest | VMVX |

For the Metal backend, IREE.gd only supports the Apple platform with Apple silicon.

Enter the [Godot Scientific Discord Server](https://discord.gg/zgSjGPDNKP) for a chat.

You can also make a feature request or bug report on the [Github issue page](https://github.com/iree-gd/iree.gd/issues).
