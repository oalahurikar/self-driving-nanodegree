# Self-Driving Car Nanodegree

**Dylan Brown**  
djwbrown _at_ gmail  
www.linkedin.com/in/djwbrown  
www.github.com/djwbrown

Welcome! Documented here are a series of projects I've built as part of the Self-Driving Car Nanodegree from Udacity.
<!-- - [Project 1 - Lane Lines](https://www.example.com) -->
<!-- - [Project 2 - Untitled](https://www.example.com) -->

## Environment setup on macOS with NVIDIA CUDA support

In this guide I'll cover installation of the following tools (updated January 2017).

1. Python 3.6
1. CUDA Toolkit 8.0 with cuDNN library
1. TensorFlow with GPU acceleration
1. OpenCV 3.2.0

**Hardware**  
Apple MacBookPro11,3  
macOS Sierra 10.12.2  
NVIDIA GeForce GT 750M

### Install the Homebrew package manager

Go to http://brew.sh for the latest instructions, or just paste the following command in the terminal.  
`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

Add the following to your `~/.bash_profile`. Make sure to re-open the terminal for these changes to take effect.
```
# Add homebrew executables to path _before_ the default path.
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
```

Make sure you have the latest version of all formulae. Use these three periodically in the future to stay up to date.
```
brew update   # Check for updates.
brew upgrade  # Use caution. May upgrade from Python 3.x to 3.y without notice.
brew doctor   # Checks that your $PATH variable and such are set correctly.
```

### Install Python 3.x

Using multiple Python installations can get complicated. My recommendation is to leave your system Python 2.7 alone, and don't mix Anaconda with Homebrew Python with Python installed from python.org. I recommend Homebrew Python based on past experience with all three. You can safely `brew install python` as well if you want 2.x alongside 3.x.

```
brew install python3
brew install python  # Optional, but recommended. Aliases pip3 to Python 3.x's pip.
pip3 install --upgrade pip setuptools wheel virtualenv
```

I'd also recommend installing jupyter notebook with
```
pip3 install notebook
```

**Testing the installation.**
```
which python       # Should print: /usr/bin/python OR /usr/local/bin/python for Homebrew Python 2.x.
python --version   # Should print: Python 2.7.13
which python3      # Should print: /usr/local/bin/python3
python3 --version  # Should print: Python 3.6.0
jupyter notebook   # Should automatically open a new browser tab. Exit cleanly with CTRL-C.
```

### Install CUDA Toolkit 8.0 and cuDNN

**NVIDIA Downloads**  
The primary CUDA tools are available for download at:  
https://developer.nvidia.com/cuda-downloads  
The neural network library cuDNN (needed for GPU acceleration in TensorFlow) is available at:  
https://developer.nvidia.com/rdp/cudnn-download

Note that membership in the "Accelerated Computing Developer Program" is required for cuDNN. Just register and the library is available gratis, even to non-students.

For the CUDA Toolkit, just follow the graphical installer. It will install to the default prefix `/Developer/NVIDIA/CUDA-8.0/`.  
For cuDNN, uncompress the tarball and copy the files to the appropriate directories.
```
sudo cp <downloaded-cuDNN-dir>/include/cudnn.h /Developer/NVIDIA/CUDA-8.0/include/
sudo cp <downloaded-cuDNN-dir>/lib/libcudnn* /Developer/NVIDIA/CUDA-8.0/lib/
sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn* /usr/local/cuda/lib/
```

Add the following to your `~/.bash_profile`, and re-open the terminal.
```
# Setting PATH for NVIDIA CUDA.
export PATH="$PATH:/usr/local/cuda/bin"
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/cuda/lib"
```

**Testing the installation.**
```
cp -r /usr/local/cuda/samples ~/cuda-samples
pushd ~/cuda-samples
make
popd

# If successful, this should output lots of useful statistics about your GPU.
~/cuda-samples/bin/x86_64/darwin/release/deviceQuery
```

### Install TensorFlow with CUDA support

The best installation documentation can be found [here](https://www.tensorflow.org/get_started/os_setup).

I will guide you through the virtualenv install. If you haven't used virtual environments before, they're a great way to keep all the installed dependencies with `pip3 install ...` separated from your system. Everything gets installed inside the virtual environment directory, and you can throw it out to start over if you break something. Again, you don't have to do things this way, but I would recommend it.

```
# Create the virtual environment in ~/tensorflow
# Use the shortened prompt header "(tf) username$".
virtualenv --system-site-packages --prompt="(tf) " ~/tensorflow
```

Add the following to your `~/.bash_profile`.
```
alias tensorflow="source ~/tensorflow/bin/activate"
```

Re-open the terminal before continuing the installation.
```
# Activate the virtual environment.
tensorflow  # You should see (tf) appear before the prompt.

# Tell pip3 where to find the TensorFlow binary distribution, built with GPU support.
# Find the fully updated list below:
# https://www.tensorflow.org/get_started/os_setup#virtualenv_installation
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.1-py3-none-any.whl

# Install TensorFlow, finally.
pip3 install --upgrade $TF_BINARY_URL
```

There is a [known issue](https://www.tensorflow.org/get_started/os_setup#mac_os_x_segmentation_fault_when_import_tensorflow) where TensorFlow looks for a library called `libcuda.1.dylib` which doesn't exist. Fix it by creating a symlink to `libcuda.dylib`, which should exist.
```
ln -sf /usr/local/cuda/lib/libcuda.dylib /usr/local/cuda/lib/libcuda.1.dylib
```

**Testing the installation.**
```
tensorflow  # Activate the virtualenv, if you haven't already.

# Start Python 3 and import tensorflow, check for errors in the log output.
python3

>>> import tensorflow as tf
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.1.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.dylib locally
>>> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:901] OS X does not support NUMA - returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GT 750M
major: 3 minor: 0 memoryClockRate (GHz) 0.9255
pciBusID 0000:01:00.0
Total memory: 2.00GiB
Free memory: 1.82GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GT 750M, pci bus id: 0000:01:00.0)
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GeForce GT 750M, pci bus id: 0000:01:00.0
I tensorflow/core/common_runtime/direct_session.cc:255] Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GeForce GT 750M, pci bus id: 0000:01:00.0

>>> print("GPU accelerated TensorFlow works!")
```

### Install OpenCV 3.x

To get information about a brew formula before installing, you can use `brew info opencv3`. I installed OpenCV 3 with the options as shown below. Only the `--with-python3` is required.
```
brew install opencv3 --with-python3 --c++11 --with-cuda --with-examples
```

There is an unusual second step needed to create the appropriate symlinks for Python to import cv2. You can preview the changes with `brew link --force --dry-run opencv3`.
```
# Avoid running anything --force without a dry-run first...
brew link --force opencv3
```

**Testing the installation.**  
Confusingly, OpenCV 3.x is still imported with `import cv2`.
```
python3

>>> import cv2
>>> cv2.__version__
'3.2.0'
```
