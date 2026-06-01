# IREE.gd with Brisklance

Since setting up a machine learning model is a one-time effort,
there are incentives to create a zoo of models that is ported to Godot
via IREE.gd.

[Brisklance](https://github.com/brisketty/brisklance) is a simple NPM-like add-on manager for Godot.
It is built to ease the workflow of add-on installation.
Unlike Godot Asset Library, it automatically downloads necessary
files only, specified by the add-on's maintainer
(instead of the whole project with manual file selection),
together with its dependencies.
This is very useful when setting up a project that uses multiple machine
learning models, as you won't need to manually download the necessary files
and assemble it together.

## Getting started

First, set up Brisklance for your existing project,
or create a Brisklance by using the Brisklance Github repository as a template.
Then, find the model that you want to install.

To install the module,

1. Go to Brisklance tab under the bottom left dock.
2. Click install.
3. Type in the Github repository and tag.
4. Click install.

And there you have it!
You can find the code under `res://addons/brisklance/plugins`.
