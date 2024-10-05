### Backend for use with Text Generation Inference

In Generative AI, the GPUs, the physical chip on a Graphic Card's PCB, steal the spotlight. While the CPU's diverse range of instructions on its Infrastructure Set Architecture can handle the tasks associated with the Operating System and many applications, it has other limitations not suitable for Generative AI. Transformer-based Neural Networks require high parallelism and abundancy, which, unlike CPUs, the Graphic Card's GPU can provide. Unlike CPUs, which are optimized for serial processing, GPUs are designed to handle the massive parallel computations required for training large neural networks. Since Generative AI relies on concurrency, like graphics cards rendering pixels on screen, the processing where one task is executed one after another, in a sequential manner, is not practical. Further, a modern CPU has 8 cores at least. A modern graphics card has between 1000 and 4000 cores. While one GPU Core cannot compare with one CPU Core, the thousands of GPU cores can accelerate matrix multiplication and other linear algebra operations, the staple of the Transformer architecture.

As the GPU capacity advances, so does the system motherboard's ability to accommodate. The system crystal oscillator provides a stable clock signal to the CPU and other components on the motherboard, including the GPU. As a simple example, this clock signal is used as a reference for Synchronous Dynamic Random Access Memory (SDRAM), which operates at the speed of the system clock. Hence, the use of the word "Synchronous". The DDR Speed Rating of SDRAM is typically twice the clock speed, and the PC Speed Rating is 8 times the DDR Speed Rating. However, with newer motherboards, the DDR Speed Rating is 4 times the Core RAM Clock Speed, and DDR3 is double DDR. The motherboard's ratings dictate what it can handle.

The CPU, with its much greater clock speed than the system crystal, relies on the external data bus to exchange data with external devices or components, such as memory modules, input/output (I/O) devices, and peripherals. When two devices with different oscillators (or two different clock domains on a GALS chip) need to transfer data to each other, they use a technique called clock domain crossing.

Conversely, the Graphics Card's GPU operates at a much higher frequency than the system clock, and it needs to communicate with the system memory and other components at a lower frequency. To achieve this, the GPU uses the same technique of clock domain crossing, which allows it to operate at a different clock frequency than the system clock.

The GPU Core Clock is the frequency at which the GPU is running measured in Hz. A Graphics Card often has a Boost Clock, the speed the card can boost itself under a heavy processing load. However, that speed drops right down if it gets too hot. This is called Thermal Throttling, the speed of the card drops down if it gets too hot. The V-RAM runs at a certain speed called the Memory Clock. This helps determine the memory bandwidth.

In fact, the GPU's clock speed is often the bottleneck in AI applications, as it determines how fast the GPU can perform matrix multiplications and other linear algebra operations. This is why NVIDIA's A100 GPUs, with their Tensor Cores and mixed-precision training capabilities, have become the standard for training large Transformer-based models.

New graphics architectures are created by AMD and NVida every 1 to 2 years. They often shrink the components of the physical GPU processor, which allows the companies to fit in more features and transistors onto the GPU Die (Die meaning the actual silicone in the chip). Architecture changes can also reduce the amount of power to run the Graphics Card. So it becomes difficult to compare different generations of graphics cards based on specs alone. Video RAM (V-RAM) serves the same function as System RAM. It holds whatever data is currently being accessed by the GPU.

### NVida A100 and H100

Parallel computing involves distributing a large task across multiple processing units, allowing each unit to execute a subtask of the task simultaneously and reducing overall processing time. The key component of NVidia's GPU architecture that enables parallel computing is CUDA cores. These small but powerful processors work together to perform calculations in parallel, greatly enhancing the overall performance of the GPU.

The CUDA core's ability to execute instructions independently of other Cuda cores allows the NVidia GPUs to process vast amounts of data in parallel, making them ideal for computationally intensive applications. This capability is particularly valuable in fields such as gaming, scientific simulations, machine learning, and video rendering, where massive amounts of data need to be processed quickly and efficiently. By leveraging the power of parallel computing, NVidia GPUs can deliver exceptional performance and accelerate complex tasks.

NVIDIA's CUDA technology has gained widespread adoption among developers due to its user-friendly interface and extensive support from software libraries and frameworks. The CUDA programming model provides developers with a flexible and intuitive way to harness the power of parallel computing on NVIDIA GPUs, making it easier to develop high-performance applications. 

Tensor Cores, on the other hand, are specialized processing units on NVIDIA GPUs that are specifically designed for matrix multiplication and other linear algebra operations. They are optimized for deep learning workloads and are designed to accelerate the performance of neural networks. Tensor Cores are capable of performing matrix multiplication operations much faster than CUDA Cores, making them ideal for tasks such as deep learning, natural language processing, and computer vision.

The NVidia A100 and H100 are intended for enterprise workloads. The Nvidia A100 was released in 2020, as part of the Ampere architecture. The Nvidia H100 was released in 2022, as part of the Hopper architecture. Here's the matchup:

Specification	    Nvidia A100 	    Nvidia H100
CUDA Cores	        6912	            14592
Tensor Cores	    432         	    576
V-RAM	            40 GB HBM2	        80 GB HBM3
Memory Bandwidth	112 GB/s	        240 GB/s
Memory Interface	512-bit	            1024-bit
Boost Clock	        1.41 GHz	        1.8 GHz
Base Clock	        1.26 GHz	        1.5 GHz
Power Consumption	250W	            350W
Architecture	    Ampere	            Hopper
Process Node	    7nm         	    5nm
GPU Die Size	    628 mm²	            814 mm²
FP32 Performance	10.5 TFLOPS     	22.4 TFLOPS
FP16 Performance	20.8 TFLOPS	        44.8 TFLOPS
INT8 Performance	41.6 TFLOPS	        89.6 TFLOPS
Tensor Performance	312 TFLOPS          624 TFLOPS (FP16)
PCIe Interface	    PCIe 4.0 x16	    PCIe 5.0 x16
NVLink	            2 x NVLink 2.0	    2 x NVLink 4.0

### Nvidia GeForce RTX 4090

The Nvidia A100 and H100 are designed for data centers. The GeForce RTX 4090 falls under NVIDIA's consumer-focused GeForce line. While primarily designed for consumer gaming, the RTX 4090 also has some capacity to run LLMs. The RTX 4090 uses the newer NVIDIA Ada Lovelace Architecture. The RTX 4090 boasts 16384 CUDA cores, 576 Tensor Cores, 24GB of V-RAM (GDDR6X), 144 Ray Tracing Cores, 14 TPS speed topped with a bus width of 384-bit along with a chip with 76.3 billion transistors. Yes, you'll be able to run Meta Llama 3 13B Instruct with this consumer Graphics Card! In comparison, the RTX 4090 is priced at $1500-$2000 compared to $10,000 – $15,000 for the NVidia A100 and $30,000+ for the NVidia H100.

### AWS EC2

EC2 Instances are organized in Instance Types. It comprises of an Instance Class, Generation and Instance Size. For example, for m5.2xlarge, the Instance Class is m, the generation is 5, and the instance size is 2xlarge. General Purpose instances are great for a diversity of workloads, a balance between compute, memory and networking. Compute Optimized instances are great for compute-intensive tasks that require high performance, such as media transcoding, High Performance Computing (HPC), machine learning and dedicated gaming servers. The Instance Class is "C". Memory Optimized instances are fast performance for workloads that process large data sets in memory, such as databases, distributed web scale cache stores, and applications performing real-time processing of big unstructured data. The Instance Class is "R". Storage Optimized Instances are great for storage-intensive tasks that require high, sequential read and write access to large data sets on local storage. An example is High frequency online transaction processing (OLTP) systems and data warehousing applications.

However, the above Instance Classes aren't the best suited for complex NLP tasks. The Amazon EC2 Service introduced the G4 instances, which are dedicated GPU instances for deploying machine learning models and graphics-intensive applications. The G4 instances are available with a choice of NVIDIA GPUs (G4dn) or AMD GPUs (G4ad). G4dn instances feature NVIDIA T4 GPUs and custom Intel Cascade Lake CPUs. Earlier, I compared the NVidia enterprise Graphics Cards, A100 and H100, and the consumer Graphics Card, GeForce RTX 4090. The NVIDIA Tesla T4 is ideal for cloud computing due to its size and features. It offers a high level of performance, boasting 2560 Cuda Cores, 320 Tensor Cores, 16 GB of V-RAM (GDDR6), 320 GB/s of Memory Bandwidth, 1.59 GHz of Boost Clock, 1.35 GHz of Base Clock, using the Turing architecture, at a lower cost than the A100 or H100 GPUs. The T4 GPU has a lower power consumption than the A100 or H100 GPUs, which makes it more suitable for datacenter deployments where power consumption is a concern. The T4 GPU has been widely available and has a well-established supply chain, which makes it easier for AWS to source and deploy in large quantities. The T4 GPU has a wide range of software support, including NVIDIA's CUDA and cuDNN libraries, which makes it easy for developers to deploy and run their applications on AWS G4 instances.  The T4 GPU is compatible with a wide range of operating systems and frameworks, including Linux, Windows, and popular deep learning frameworks like TensorFlow and PyTorch.

Note in March of 2023, AWS announced a collaboration with NVIDIA for the EC2 P5 Instance, which is powered by the NVidia H100 Tensor Core GPUs for accelerated generative AI and HPC applications. Also, note that the EC2 P4d instances are backed by the A100s. This is important as the Turing architecture has been somewhat deprecated in a fast evolving AI industry. For example, the HuggingFace TGI Inference uses Flash Attention, a technique that reduces the computational cost of attention mechanisms in transformers by using a more efficient algorithm. It does not support the older Turing architecture. As a runtime error I received succintly states it: "RuntimeError: FlashAttention only supports Ampere GPUs or newer." 

The NVIDIA A10G Tensor Core GPU is powered by the NVIDIA Ampere Architecture. Beyond that, it boasts more capacity than the Tesla T4s of the Turing architecture. The A10G has 6928 CUDA Cores, 288 Tensor Cores, 24GB of GDDR6 V-RAM (Video RAM or GPU memory), 384-bit memory bus, 600.2 GB/s memory bandwidth, 1695 MHz Boost Clock, 885 MHz Base Clock, and 28,300 million transistors. 

### AWS SageMaker

To be added soon