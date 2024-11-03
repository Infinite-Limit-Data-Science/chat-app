### AWS EC2 Instance Types (Class, Generation, and Instance Size)

EC2 Instances are organized in Instance Types. It comprises of an Instance Class, Generation and Instance Size. For example, for m5.2xlarge, the Instance Class is m, the generation is 5, and the instance size is 2xlarge. General Purpose instances are great for a diversity of workloads, a balance between compute, memory and networking. Compute Optimized instances are great for compute-intensive tasks that require high performance, such as media transcoding, High Performance Computing (HPC), machine learning and dedicated gaming servers. The Instance Class is "C". Memory Optimized instances are fast performance for workloads that process large data sets in memory, such as databases, distributed web scale cache stores, and applications performing real-time processing of big unstructured data. The Instance Class is "R". Storage Optimized Instances are great for storage-intensive tasks that require high, sequential read and write access to large data sets on local storage. An example is High frequency online transaction processing (OLTP) systems and data warehousing applications.

However, the above Instance Classes aren't the best suited for complex NLP tasks. The Amazon EC2 Service introduced the G4 instances, which are dedicated GPU instances for deploying machine learning models and graphics-intensive applications. The G4 instances are available with a choice of NVIDIA GPUs (G4dn) or AMD GPUs (G4ad). G4dn instances feature NVIDIA T4 GPUs and custom Intel Cascade Lake CPUs. Earlier, I compared the NVidia enterprise Graphics Cards, A100 and H100, and the consumer Graphics Card, GeForce RTX 4090. The NVIDIA Tesla T4 is ideal for cloud computing due to its size and features. It offers a high level of performance, boasting 2560 Cuda Cores, 320 Tensor Cores, 16 GB of V-RAM (GDDR6), 320 GB/s of Memory Bandwidth, 1.59 GHz of Boost Clock, 1.35 GHz of Base Clock, using the Turing architecture, at a lower cost than the A100 or H100 GPUs. The T4 GPU has a lower power consumption than the A100 or H100 GPUs, which makes it more suitable for datacenter deployments where power consumption is a concern. The T4 GPU has been widely available and has a well-established supply chain, which makes it easier for AWS to source and deploy in large quantities. The T4 GPU has a wide range of software support, including NVIDIA's CUDA and cuDNN libraries, which makes it easy for developers to deploy and run their applications on AWS G4 instances.  The T4 GPU is compatible with a wide range of operating systems and frameworks, including Linux, Windows, and popular deep learning frameworks like TensorFlow and PyTorch.

Note in March of 2023, AWS announced a collaboration with NVIDIA for the EC2 P5 Instance, which is powered by the NVidia H100 Tensor Core GPUs for accelerated generative AI and HPC applications. Also, note that the EC2 P4d instances are backed by the A100s. This is important as the Turing architecture has been somewhat deprecated in a fast evolving AI industry. For example, the HuggingFace TGI Inference uses Flash Attention, a technique that reduces the computational cost of attention mechanisms in transformers by using a more efficient algorithm. It does not support the older Turing architecture. As a runtime error I received succintly states it: "RuntimeError: FlashAttention only supports Ampere GPUs or newer." 

The NVIDIA A10G Tensor Core GPU is powered by the NVIDIA Ampere Architecture. Beyond that, it boasts more capacity than the Tesla T4s of the Turing architecture. The A10G has 6928 CUDA Cores, 288 Tensor Cores, 24GB of GDDR6 V-RAM (Video RAM or GPU memory), 384-bit memory bus, 600.2 GB/s memory bandwidth, 1695 MHz Boost Clock, 885 MHz Base Clock, and 28,300 million transistors. 

### AWS EC2 Launch Templates and Spot Instances  

A Launch Template contains information on how to launch ec2 instances within your ASG. You specify **AMI, Instance Type, EC2 User Data, EBS Volume, Security Groups, SSH Key Pairs, IAM Roles for EC2 Instances, Subnet information, and Load Balancer.** For example, to run Meta Llama 3.2 90B Vision Instruct, it will require 192GB of VRAM. The most cost-efficient NVidia GPU is the A10G. EC2 supports A10Gs through the G5 (G Instance Class and Generation 5) instance types. The g5.48xlarge instance type boasts 192GiB of VRAM among 8 A10G GPUs. It also supports 192GiB of CPU and 768GiB of memory. In addition to the Instance Type, the AMI must also be specified. For Machine Learning in EC2, a common choice is the AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1 (Ubuntu 20.04) 20240923. An instance must have a security group attached to an ENI, and as such, any instance that launches from this launch template will require a security group. So define the security group in the launch template. Equally, you will want to be able to connect to instances so provide a SSH Key Pair if necessary. Finally, define the ebs root volume. This is critical: assign at least 2000GiB to EBS volume for a large model like meta llama 3.2 90B Vision Instruct, given there will be a lot of model files and cache. Set volume type to GP3 so you get minimum 3000 IOPS and 125mb throughput. For Purchasing Options, set to SPOT INSTANCES! Special note: Spot Instances don't failover to on-demand. In order to accomplish this, you need to use CloudWatch Alarms. Finally, create the Launch Template.

### AWS AMI

An AMI is a customization of an EC2 Instance. You add your own software, configuration, operating system. It leads to faster boot because all your software is pre-packaged. **AMIs are built for a specific region. Not an AZ but a region!** You can launch EC2 instances from a public AMI, Marketplace AMI (an AMI someone else) or a custom AMI you create and maintain. The AMI process includes: 1) start an EC2 Instance and customize it, 2) Stop the Instance, 3) Build the AMI (this will also create EBS snapshots), 4) Launch instances from the AMI.

A common practice is to use a **Golden AMI** to install your applications, OS dependencies, etc beforehand and launch EC2 instances from the Golden AMI. For dynamic configuration, use User Data scripts to bootstrap instances.

As already mentioned, for Machine Learning in EC2, a common choice is the AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1 (Ubuntu 20.04) 20240923.

### AWS EC2 Auto Scaling Groups

Now you must create an Auto Scaling Group.

Auto-Scaling Groups allow you to scale out (add EC2 instances) to match an increased load and scale in (remove EC2 instances) to match a decreased load. **You can automatically register new instances to a load balancer. ELB can check health of your instances.** You must set a minimum capacity, desired capacity and maximum capacity.

Auto-Scaling Groups support Scaling Policies. You can scale an ASG based on CloudWatch alarms. An Alarm monitors a metric, such as Average CPU or a custom metric. Based on the alarm, we can create scale-out policies and scale-in policies. Auto Scaling Group's Scaling Policies include Dynamic Scaling, Scheduled Scaling and Predictive Scaling.

Dynamic Scaling supports Target Tracking Scaling, Simple and Step Scaling. **Target Tracking Scaling allows you to specify an average, such as the average CPU to stay around 40 percent.** With Step Scaling, you define steps, such as when an alarm is triggered (CPU > 70 percent), then add 2 units; and when an alarm is triggered (CPU < 30 percent), then remove 1 unit. **Scheduled Scaling allows you to anticipate a scaling based on known usage patterns, e.g. increase the min capacity to 5pm on Fridays.** Predictive Scaling is another good options where you can forecast load and schedule scaling in advance. Good metrics to scale on include CPUUtilization, RequestCountPerTarget, Average Network In/Out and any custom metric.

Note after a scaling activity happens (scale in or scale out), you are in a cool down period (default 300 seconds). During the cool down period, the ASG will not launch or terminate additional instances (to allow for metrics to stabilize). It is recommended to use a ready-to-use AMI to reduce configuration time in order to be serving requests faster.

We will configure an Auto Scaling Group because we want to spin up an EC2 instance that meets the Spot price in any AZ in the given region. Different AZs have different capacity, and there is a possibility a g5.48xlarge instance type is available at the Spot price in one AZ but not at another. Hence, if the Spot price is not met in one AZ, the instance will stop, but then it can have an opportunity to run in another AZ when the Spot price is met. Under the Network section in Auto Scaling Groups console, make sure to select all AZs under "Availability zones and subnets". We will not be using a Load Balancer for our instance so select "No load balancer". Set the Desired Capacity to 1, since we will only want to run 1 instance at any given time. Also, we set the Min desired capacity to 1 and Max desired capacity to 1 as we only want 1 instance running at any given time. 

No need to add Scaling Policy. Instance Maintenance Policy set to "Launch before terminating". The underlying data will be preserved because all the instances will use the same EBS volume!!!!

Once the EC2 Instance is launched, name the EC2 Instance "gpu192-ml" to indicate it is the machine with 192GiB of GPU memory.

### Spot Requests

EC2 Spot Instance Requests can get you a discount of up to 90 percent compared to On-demand. Define the max spot price and get the instance while the current spot price is less than the max spot price. **If the current spot price is greater than your max spot price, you can choose to Stop or Terminate your instance within a 2 minute grace period. You can decide on a Request Type of either one-time or persistent.** If its a one-time Request for a Spot Instance, as soon as Request is fulfilled, your instances will be launched and your Spot Request will go away. **With a Persistent Request Type, we want our instances to be valid as long as the Valid From and Valid Until parameters are met. If you want to cancel a Spot Request, it needs to be in the Open, Active or Disabled state. Cancelling a Spot Request does not terminate instances. To terminate Spot Instances, you need to first cancel the Spot Request and then terminate the Spot Instances.**

A Spot Fleet is a set of Spot Instances with optional On-demand Instances. You can define multiple Launch Pools so the fleet can choose. Strategies to allocate Spot Instances include lowestPrice, diversified, capacityOptimized, and priceCapacityOptimized. priceCapacityOptimized is recommended since it selects pools with the highest capacity and then selects the pool with the lowest price.

### ENI

AN ENI represents a Virtual Network Card. **An EC2 Instance can have one or more ENI (e.g. Eth0, Eth1). The ENI can have one primary private IPv4 and one or more secondary IPv4. It can have one IPv4 Elastic IP per private IPv4. It can have one Public IPv4. You can have one ore more Security Groups attached to the ENI.** You can have a Mac Address attached to the ENI. You can create ENIs independently in the EC2 Console and then attach them to an EC2 Instance. You can attach them on the fly, moving them from one EC2 Instance to another for failover. Importantly, the ENI is bound to a specific AZ.

### Elastic IP

An Elastic IP (EIP) is a static, public IPv4 address that you can allocate to your AWS account and use to dynamically associate with any instance or network interface in a given region. 

We will want to use an ELP because when using Spot Instances, the instance will stop when it no longer meets the Spot max price. It will relaunch again as part of the Auto Scaling Group, but with a different public IP address.

### Run Container in EC2
model=meta-llama/Llama-3.2-90B-Vision-Instruct
token=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
volume=$PWD/data
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.4.0 --model-id $model --max-batch-prefill-tokens 12582 --max-batch-total-tokens 16777 --max-input-tokens 12582 --max-total-tokens 16777 --max-concurrent-requests 1 --num-shard 8 --quantize bitsandbytes-nf4





sudo vi /etc/systemd/system/llama3-1-70b-instruct
[Unit]
Description=meta-llama/Llama-3.1-70B-Instruct on HuggingFace TGI
After=docker.service
Requires=docker.service
BindsTo=docker.service
ConditionPathExists=/home/ubuntu/data/models--hugging-quants--Meta-Llama-3.1-70B-Instruct-AWQ-INT4

[Service]
ExecStart=/usr/bin/docker run --gpus all --shm-size 1g \
  -e HUGGING_FACE_HUB_TOKEN=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 \
  -p 8080:80 \
  -v /home/ubuntu/data:/data \
  ghcr.io/huggingface/text-generation-inference:2.3.1 \
  --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --max-input-length 4768 \
  --max-total-tokens 10960 \
  --num-shard 4
ExecStop=/usr/bin/docker stop %n
ExecReload=/usr/bin/docker restart %n
Restart=always
RestartSec=10
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target

sudo systemctl daemon-reload

systemctl status llama3-1-70b-instruct.service
● llama3-1-70b-instruct.service - meta-llama/Llama-3.1-70B-Instruct on HuggingFace TGI
     Loaded: loaded (/etc/systemd/system/llama3-1-70b-instruct.service; disabled; vendor preset: enabled)
     Active: inactive (dead)

sudo systemctl start llama3-1-70b-instruct.service

sudo systemctl status llama3-1-70b-instruct.service

sudo systemctl enable llama3-1-70b-instruct.service
 Created symlink /etc/systemd/system/multi-user.target.wants/llama3-1-70b-instruct.service → /etc/systemd/system/llama3-1-70b-instruct.service.


# enabled must it will automatically start at boot
systemctl list-unit-files --type=service | grep llama3-1-70b-instruct.service
  llama3-1-70b-instruct.service                  enabled         enabled


### AWS SageMaker

To be added soon