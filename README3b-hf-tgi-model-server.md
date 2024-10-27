### Text Generation Inference Architecture: The Model Server

**The model server is a python server, capable of starting a server waiting for gRPC requests, loads a given model, perform sharding to provide tensor parallelism, and stays alive while waiting for new requests. The model server supports models instantiated using Pytorch and optimized for inference mainly on CUDA/ROCM**. By default, the model server will attempt building a server optimized for Nvidia GPUs with CUDA.

### NVidia Cuda Primer

NVIDIA CUDA is a parallel computing platform and programming model that allows developers to harness the power of GPUs (Graphics Processing Units) for general-purpose computing tasks.

The development of CUDA was driven by the need to overcome the performance limitations of traditional CPUs for certain types of computational tasks. GPUs, with their highly parallel architecture consisting of hundreds or even thousands of cores, proved to be well-suited for these types of workloads.

Initially, CUDA was primarily used in fields such as scientific research and high-performance computing. However, it quickly gained popularity across various industries due to its ability to accelerate computationally intensive applications.

**Traditional CPUs (Central Processing Units) are designed for sequential processing, executing one task at a time. In contrast, GPUs consist of thousands of smaller cores that can execute multiple tasks simultaneously. This makes them highly efficient for performing computationally intensive operations**.

**When using CUDA, developers write code using C or C++ programming languages along with special extensions provided by NVIDIA**. The code is then compiled specifically for execution on GPUs.

The key concept behind GPU parallel computing with CUDA is dividing large computational tasks into smaller subtasks that can be executed concurrently on different GPU cores. This allows for massive acceleration in performance compared to traditional CPU-based computations.

By utilizing the power of many GPU cores working together in parallel, complex calculations can be completed much faster than would be possible with CPUs alone. This makes CUDA ideal for applications such as scientific simulations, data analysis, machine learning algorithms, and even video game development.

In the field of healthcare, CUDA plays a crucial role in accelerating medical research and diagnostics. It enables researchers to process large amounts of genomic data quickly, leading to advancements in personalized medicine and targeted therapies. Additionally, CUDA is utilized for medical imaging tasks such as CT scans or MRI analysis, enabling faster image reconstruction and more accurate diagnoses.

**You will need an NVIDIA GPU that supports CUDA. Check the specifications of your GPU to ensure it has CUDA cores, as these are essential for running parallel computations on the GPU. The more CUDA cores your GPU has, the more processing power you can leverage.**

Once you have confirmed that your hardware meets the requirements, it’s time to install the necessary tools. Start by downloading and installing the latest version of NVIDIA CUDA Toolkit from their official website. This toolkit includes everything you need to develop and run CUDA applications.

After installing the toolkit, don’t forget to update your graphics drivers to ensure optimal performance and compatibility with CUDA.

To test if everything is set up correctly, try running some sample codes provided in the CUDA Toolkit documentation or online tutorials. These samples will help you understand how parallel computing works using GPUs.

### Text Generation Inference Architecture: The Model Server CLI

The official command line interface (CLI) for the server supports three subcommands, download-weights, quantize and serve:

- download-weights will download weights from the hub and, in some variants it will convert weights to a format that is adapted to the given implementation;
- quantize will allow to quantize a model using the qptq package. This feature is not available nor supported on all variants;
- **serve will start the server that load a model (or a model shard), receives gRPC calls from the router, performs an inference and provides a formatted response to the given request**.

Once both components are initialized, weights downloaded and model server is up and running, router and model server exchange data and info through the gRPC call.

