### Text Generation Inference Architecture: The Model Server

**The model server is a python server, capable of starting a server waiting for gRPC requests, loads a given model, perform sharding to provide tensor parallelism, and stays alive while waiting for new requests. The model server supports models instantiated using Pytorch and optimized for inference mainly on CUDA/ROCM**. By default, the model server will attempt building a server optimized for Nvidia GPUs with CUDA.

### NVidia CUDA Toolkit

CUDA is a parallel computing platform and programming model invented by NVIDIA. It enables dramatic increases in computing performance by harnessing the power of the graphics processing unit (GPU).

CUDA was developed with several design goals in mind:

- **Provide a small set of extensions to standard programming languages, like C, that enable a straightforward implementation of parallel algorithms. With CUDA C/C++, programmers can focus on the task of parallelization of the algorithms rather than spending time on their implementation**.
- **Support heterogeneous computation where applications use both the CPU and GPU. Serial portions of applications are run on the CPU, and parallel portions are offloaded to the GPU**. As such, CUDA can be incrementally applied to existing applications. The CPU and GPU are treated as separate devices that have their own memory spaces. This configuration also allows simultaneous computation on the CPU and GPU without contention for memory resources.
- CUDA-capable GPUs have hundreds of cores that can collectively run thousands of computing threads. These cores have shared resources including a register file and a shared memory. **The on-chip shared memory allows parallel tasks running on these cores to share data without sending it over the system memory bus**.

To use NVIDIA CUDA on your system, you will need the following installed:

- CUDA-capable GPU
- A supported version of Linux with a gcc compiler and toolchain
- CUDA Toolkit

**The CUDA development environment relies on tight integration with the host development environment, including the host compiler and C runtime libraries**, and is therefore only supported on distribution versions that have been qualified for this CUDA Toolkit release.

For example, CUDA supports the following Ubuntu distributions, Kernel, GCC, GLIBC:

Distribution                            Kernel      Default GCC         GLIBC
Ubuntu 24.04.z (z <= 1) LTS (x86_64)    6.8.0-41    13.2.0              2.39
Ubuntu 22.04.z (z <= 4) LTS (x86_64)    6.5.0-27    12.3.0              2.35
Ubuntu 20.04.z (z <= 6) LTS (x86_64)    5.15.0-67   9.4.0               2.31

GCC is a compiler system produced by the GNU Project that supports various programming languages, primarily C and C++. It is widely used for compiling and building software on Linux systems.

The version of GCC is critical because CUDA and other software frameworks rely on specific features and compatibility provided by certain versions of the compiler. Using a supported GCC version ensures that the software compiles and runs as expected without encountering compatibility issues or errors.

GLIBC is the GNU implementation of the C standard library, providing core functionalities such as system calls, basic functions like string handling, memory management, and I/O operations.

The version of GLIBC is crucial because many software applications, including CUDA, rely on its functions to interact with the underlying system and manage resources. An unsupported version of GLIBC can lead to compatibility issues, preventing applications from running properly. GLIBC version changes may introduce new features or deprecate old ones, affecting how programs behave.

The kernel is the core part of the Linux operating system that manages system resources, hardware interactions, and processes. It acts as an intermediary between software applications and the physical hardware.

The kernel version is important for CUDA compatibility because it influences driver support, system stability, and the underlying mechanisms that CUDA depends on for GPU management and operations. Specific kernel versions ensure that CUDA and its drivers have the necessary modules and APIs for effective GPU utilization. Newer kernel versions often come with improvements in hardware support, performance enhancements, and security updates, which are critical for running high-performance applications like those that use CUDA.

For example, on the Linux Operating System,on the x86_64 architecture, on the Ubuntu distribution, version 24.04, using the deb (local) installer type, this is how you can install the CUDA Toolkit using a local installer:

```shell
# Downloads a pin file to ensure the CUDA repository has a high priority during package installation, preventing conflicts with other repositories.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin

# Moves the pin file to the appropriate directory to set the priority for the CUDA repository.
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Downloads a local installer package for CUDA. This is a .deb package specific to the version and distribution.
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb

# Installs the local CUDA repository package on the system.
# The dpkg command is a low-level tool used for managing Debian packages on Debian-based systems like Ubuntu
# cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb is the .deb package file for the CUDA repository. It includes information and configurations that set up the local CUDA repository on your system. 
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb

# Copies the GPG keyring for repository authentication to the appropriate directory.
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Updates the package list and installs the specified version of CUDA Toolkit (in this case, 12.6)
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# NVIDIA Driver Instructions:
sudo apt-get install -y nvidia-open
```

When you run the dpkg command, dpkg installs the .deb package onto your system. Specifically, it does the following:

- Extracts the package: Unpacks the content of the .deb file and places it in the appropriate system directories.
- Configures the package: Runs any post-installation scripts included in the package to configure the package properly.
- Registers the package: Updates the package database to include the new package, making it available for apt to manage in future operations.

This .deb package sets up a local repository on your system that points to CUDA-related packages, allowing you to install CUDA and its dependencies using apt or apt-get.  Often, you need to ensure that the repository is trusted, so a GPG key step (sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/) is included to validate the authenticity of packages from this repository. The .deb package typically sets up the necessary apt configuration files (e.g., in /etc/apt/sources.list.d/) so that apt can recognize and access the CUDA packages for installation.

dpkg provides more direct control over package installation compared to higher-level tools like apt. It allows you to install packages downloaded manually or from a source outside the usual repositories. Installing a .deb package directly with dpkg does not require an active internet connection, which is useful for offline installations. Unlike apt, dpkg does not resolve dependencies automatically. If the package requires other packages to be installed, dpkg will not handle that for you, and you might need to run:

```shell
# This command fixes and installs any missing dependencies after using dpkg.
sudo apt-get install -f
```

Difference Between apt and dpkg:

apt (Advanced Package Tool):

- High-level tool that simplifies the process of managing packages.
- Handles package installation, removal, and dependency resolution automatically.
- Connects to online repositories or local repositories configured on your system.

dpkg (Debian Package Manager):

- Low-level tool for managing .deb packages.
- Installs .deb packages directly without connecting to a repository.
- Does not handle dependencies automatically; if a package has unmet dependencies, you need to resolve them separately.

For setting up a new local repository (as in the case of NVIDIA's CUDA repository setup), dpkg installs the repository configuration package, allowing apt to recognize and pull packages from that local or custom repository. apt relies on a configured repository index to find and install packages. If you have a standalone .deb package, apt won’t install it unless that package is part of a repository that apt knows about.

apt installs packages from sources defined in your /etc/apt/sources.list or /etc/apt/sources.list.d/ files. If the package isn’t listed in those sources, apt won’t find or install it.

Note there is a second set of instructions to install CUDA. This set installs CUDA by setting up the CUDA repository directly from NVIDIA's servers and then installing it via apt:

```shell
# Installs the CUDA repository package directly.
sudo dpkg --install cuda-repo-<distro>-<version>.<architecture>.deb

# Deletes an existing GPG key, likely to remove an outdated key for repository authentication.
sudo apt-key del 7fa2af80

# Downloads the new CUDA keyring package. 
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<arch>/cuda-keyring_1.1-1_all.deb

# Adds the contrib repository to enable additional packages, which may be necessary for some dependencies.
sudo add-apt-repository contrib

# Updates the package list and installs the latest available version of CUDA from the configured repository.
sudo apt-get update
sudo apt-get -y install cuda
```

This approach sets up the repository directly from NVIDIA's server, allowing for easier updates and management. By using sudo apt-get install cuda, it installs the latest version available in the configured repository.

The first method uses a local installer (.deb package) to set up the repository. The second method directly configures the repository from NVIDIA's online source. The first method allows you to install a specific version by specifying the local installer version. The second method installs the latest version available unless you specify a particular package version. The second method includes additional steps for updating the GPG keyring for better security practices. Using the second method allows for easier updates in the future as it is directly connected to NVIDIA's repository.

Which Method to Use? First Method: Use this if you need to install a specific version of CUDA and prefer a local installer approach. Second Method: Use this if you want a more streamlined setup that connects directly to NVIDIA's repository and allows for easier version updates.

The next set of instructions must be used if you are installing in a WSL environment. Do not use the Ubuntu instructions in this case.

```shell
# Install repository meta-data
sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb

# Update the CUDA public GPG key
sudo apt-key del 7fa2af80

# When installing using the local repo:
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

# When installing using the network repo:
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<arch>/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Pin file to prioritize CUDA repository:
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<architecture>/cuda-<distro>.pin
sudo mv cuda-<distro>.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Update the Apt repository cache and install CUDA
sudo apt-get update
sudo apt-get install cuda
```

### Why CUDA Toolkit 

**Traditional CPUs (Central Processing Units) are designed for sequential processing, executing one task at a time. In contrast, GPUs consist of thousands of smaller cores that can execute multiple tasks simultaneously. This makes them highly efficient for performing computationally intensive operations**.

**When using CUDA, developers write code using C or C++ programming languages along with special extensions provided by NVIDIA**. The code is then compiled specifically for execution on GPUs.

The key concept behind GPU parallel computing with CUDA is dividing large computational tasks into smaller subtasks that can be executed concurrently on different GPU cores. This allows for massive acceleration in performance compared to traditional CPU-based computations.

By utilizing the power of many GPU cores working together in parallel, complex calculations can be completed much faster than would be possible with CPUs alone. This makes CUDA ideal for applications such as scientific simulations, data analysis, machine learning algorithms, and even video game development.

In the field of healthcare, CUDA plays a crucial role in accelerating medical research and diagnostics. It enables researchers to process large amounts of genomic data quickly, leading to advancements in personalized medicine and targeted therapies. Additionally, CUDA is utilized for medical imaging tasks such as CT scans or MRI analysis, enabling faster image reconstruction and more accurate diagnoses.

**You will need an NVIDIA GPU that supports CUDA. Check the specifications of your GPU to ensure it has CUDA cores, as these are essential for running parallel computations on the GPU. The more CUDA cores your GPU has, the more processing power you can leverage.** Note the NVidia A10G, A100, and, of course, H100 all contain CUDA cores.

### Text Generation Inference Architecture: The Model Server CLI

The official command line interface (CLI) for the server supports three subcommands, download-weights, quantize and serve:

- download-weights will download weights from the hub and, in some variants it will convert weights to a format that is adapted to the given implementation;
- quantize will allow to quantize a model using the qptq package. This feature is not available nor supported on all variants;
- **serve will start the server that load a model (or a model shard), receives gRPC calls from the router, performs an inference and provides a formatted response to the given request**.

Once both components are initialized, weights downloaded and model server is up and running, router and model server exchange data and info through the gRPC call.

