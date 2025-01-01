# Foreword
This repo was created for anyone curious as to what is going on with Intel GPUs. There has been a lot of shenanigans with Intel recently regarding their GPUs and I decided to purchase a few.

In total I have four (SPARKLE Intel Arc A770 TITAN OC Edition 16GB) gpus of which I plan on building a multi-gpu machine later in 2025 for my Masters in Software Engineering at PSU Capstone Class.

The GPU referred to in this tutorial is an Intel A770 Sparkle 16 GB. I originally purchased my first card of Newegg for $269.99. 

Here are the the GPU specifications.


| **Technical Specifications**              | **Details**                                      |
|-------------------------------------------|--------------------------------------------------|
| **GPU**                           | Intel® Arc™ A770                          |
| **Architecture**                  | Xe HPG (Alchemist)                        |
| **Process Technology**            | 6 nm                                      |
| **Execution Units**               | 512                                       |
| **Shading Units (Cores)**         | 4096                                      |
| **Ray Tracing Cores**             | 32                                        |
| **Base Clock**                    | 2100 MHz                                  |
| **Boost Clock**                   | 2400 MHz                                  |
| **Memory Capacity**               | 16 GB GDDR6                               |
| **Memory Interface**              | 256-bit                                   |
| **Memory Speed**                  | 17.5 Gbps                                 |
| **Memory Bandwidth**              | 560 GB/s                                  |
| **Cooling Solution**              | TORN Cooling with triple AXL fans         |
| **Slot Design**                   | 2.5-slot                                  |
| **Backplate**                     | Full-metal, pre-installed                 |
| **Dimensions**                    | 306 mm (L) x 103 mm (H)                   |
| **Power Connectors**              | 2 x 8-pin                                 |
| **TDP**                           | 225 W                                     |
| **Recommended PSU**               | 650 W or greater                          |
| **Display Outputs**               | 1 x HDMI 2.0b, 3 x DisplayPort 2.0        |
| **Maximum Resolution**            | 7680 x 4320 (8K)                          |
| **DirectX Support**               | DirectX 12 Ultimate                       |
| **Vulkan Support**                | Vulkan 1.3                                |
| **OpenGL Support**                | OpenGL 4.6                                |
| **OpenCL Support**                | OpenCL 3.0                                |
| **Adaptive Sync**                 | Supported                                 |
| **Ray Tracing**                   | Hardware-accelerated                      |
| **Variable Rate Shading (VRS)**   | Supported                                 |
| **AV1 Encode/Decode**             | Supported                                 |
| **VP9 Bitstream & Decoding**      | Supported                                 |
| **Intel® Deep Link Technologies** | Hyper Compute, Hyper Encode, Stream Assist|
| **Additional Features**           | ThermalSync LED with temperature-based    |
|                                   | color change                              |

# A special thanks to
This guide was kicked off by me finding [Christian Mills article](https://christianjmills.com/posts/intel-pytorch-extension-tutorial/native-ubuntu/) on getting the GPU online.

In this tutorial, I’ll guide you through setting up Intel’s [PyTorch extension](https://github.com/intel/intel-extension-for-pytorch) on [Ubuntu](https://ubuntu.com/download/desktop) to train models with their Arc GPUs. The extension provides Intel’s latest feature optimizations and hardware support before they get added to PyTorch. Most importantly for our case, it includes support for Intel’s Arc GPUs and optimizations to take advantage of their Xe Matrix Extensions (XMX).

The XMX engines are dedicated hardware for performing matrix operations like those in deep-learning workloads. Intel’s PyTorch extension allows us to leverage this hardware with minimal changes to existing PyTorch code.

To illustrate this, we will walk through two use cases. First, we’ll adapt the training code from my beginner-level PyTorch tutorial, where we fine-tune an image classification model from the [timm](https://github.com/huggingface/pytorch-image-models) library for hand gesture recognition. Second, we will use the ipex-llm library, which builds on Intel’s PyTorch extension, to perform inference with the [LLaMA 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) model.

# Introduction to Intel OneAPI (currently using 2025.0)
Intel's oneAPI is an open, unified programming model designed to simplify development across diverse computing architectures, including CPUs, GPUs, FPGAs, and other accelerators. It provides a comprehensive set of tools and libraries that enable developers to create high-performance, cross-architecture applications without the need to manage multiple codebases. In relation to deep learning frameworks, Intel offers extensions for [TensorFlow](https://intel.github.io/intel-extension-for-tensorflow/latest/get_started.html) and [PyTorch](https://intel.github.io/intel-extension-for-pytorch/) that integrate with oneAPI to optimize performance on Intel hardware. The Intel® Extension for TensorFlow* is a high-performance plugin that brings Intel CPU and GPU devices into the TensorFlow ecosystem, facilitating AI workload acceleration. Similarly, the Intel® Extension for PyTorch enhances deep learning training and inference performance on Intel processors by extending PyTorch with the latest optimizations.

[YouTube Video](https://youtu.be/qvxmeupPftU)

Intel provides a comprehensive collection of oneAPI [code samples](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples) designed to help developers understand and implement various features of the oneAPI toolkits. These samples cover a wide range of topics, including data parallel programming, artificial intelligence, and high-performance computing, offering practical examples for both beginners and experienced developers. Each sample includes detailed instructions and code snippets to facilitate learning and application development. Here are some notable examples:

    * Intel® Extension for TensorFlow Getting Started Sample*: This sample guides users on running a TensorFlow* inference workload on both GPU and CPU using Intel® AI Tools, and demonstrates how to analyze GPU and CPU usage via oneDNN verbose logs.
    * Intel® Neural Compressor Sample for PyTorch: This sample demonstrates how to perform INT8 quantization on a Hugging Face BERT model using the Intel® Neural Compressor, showcasing inference optimization techniques. 
    * Intel® oneAPI Math Kernel Library (oneMKL) Fast Fourier Transform (FFT) Sample: This sample illustrates how to implement FFT computations using oneMKL, highlighting performance optimizations for mathematical computations.
    * Intel® oneAPI Data Parallel C++ (DPC++) FPGA Design Example: This sample provides a walkthrough of developing FPGA applications using DPC++, demonstrating efficient hardware acceleration techniques.
    * Intel® oneAPI Video Processing Library (oneVPL) Decode and Encode Sample: This sample showcases how to utilize oneVPL for real-time video decoding and encoding, emphasizing media processing capabilities.

# Build
## Case
[LINK](https://www.newegg.com/black-jonsbo-mesh-screen-mtx-case-micro-atx/p/2AM-006A-000B8?Item=9SIAY3SJNH9664) JONSBO D31 MESH SC BLACK Micro ATX Computer Case with Sub HD-LCD Display, M-ATX/DTX/ITX Mainboard/Support RTX 4090(335-400mm) GPU 360/280AIO,Power ATX/SFX: 100mm-220mm Multiple Tool-free Design,Black

## Cooling Accessories
### AIO
[LINK](https://www.newegg.com/be-quiet-liquid-cooling-system/p/2YM-0069-00004?Item=9SIA68VBZU7750) be quiet! PURE LOOP 360mm AIO CPU Water Cooler | All In One Water Cooling System | Intel and AMD Support | Low Noise | BW008

[LINK](https://www.newegg.com/icegale-fixed-case-fan/p/1YF-0184-00053?Item=9SIAT5SJH17984) Iceberg Thermal IceGALE Silent 120mm (Black) 3-PACK Quiet Optimized Airflow 3-Pin Case Fans

[LINK](https://www.newegg.com/bgears-vortex-120-case-fan/p/N82E16835132054?Item=N82E16835132054) Bgears Vortex 120 ARGB LED Case Fans b-ARGB Vortex 120

[LINK](https://www.newegg.com/thermal-grizzly/p/13C-003E-00004?Item=9SIA4RE6M28090) Thermal Grizzly Kryonaut Thermal Grease Paste - 1.0 Gram

## CPU
[LINK](https://www.newegg.com/amd-ryzen-9-5900x/p/N82E16819113664?Item=N82E16819113664) AMD Ryzen 9 5900X - Ryzen 9 5000 Series Vermeer (Zen 3) 12-Core 3.7 GHz Socket AM4 105W None Integrated Graphics Desktop Processor - 100-100000061WOF

## GPU
[LINK](https://www.newegg.com/sparkle-arc-a770-sa770t-16goc/p/N82E16814993004?Item=N82E16814993004) SPARKLE Intel Arc A770 TITAN OC Edition, 16GB GDDR6, ThermalSync, TORN Cooling, Axial Fan, Metal Backplate, SA770T-16GOC

## Hard Drive(s)
### Ubuntu Linux Server
[LINK](https://www.samsung.com/us/computing/memory-storage/solid-state-drives/990-pro-w-heatsink-pcie-4-0-nvme-ssd-1tb-mz-v9p1t0cw/?cid=pla-ecomm-pfs-mms-us-google-na-03042022-170119-&ds_e=GOOGLE-cr:0-pl:267548417-&ds_c=FF~Memory+PMax_CN~Memory+PMax_ID~B0000PWD_PH~on_MK~us_BS~ms_PR~ecom_SB~memcross_FS~hqloe_CA~smp_KS~nag_MT~na-&ds_ag=-&ds_k=&gad_source=1&gclid=CjwKCAiA1fqrBhA1EiwAMU5m_zGVd5S4llCHabWb0dBGGrnaAD4y-z5oO5w4VcCuGRn1VwT-98lt_RoC5osQAvD_BwE&gclsrc=aw.ds) 990 PRO w/ Heatsink PCIe®4.0 NVMe™ SSD 1TB

### Ubuntu Linux
[LINK](https://www.newegg.com/samsung-970-evo-plus-500gb/p/N82E16820147742?Item=N82E16820147742) SAMSUNG 970 EVO PLUS M.2 2280 500GB PCIe Gen 3.0 x4, NVMe 1.3 V-NAND Internal Solid State Drive (SSD) MZ-V7S500B/AM

## Memory
[LINK](https://www.newegg.com/g-skill-128gb-288-pin-ddr4-sdram/p/N82E16820232992?Item=N82E16820232992) G.SKILL TridentZ RGB Series 128GB (4 x 32GB) 288-Pin PC RAM DDR4 3600 (PC4 28800) Desktop Memory Model F4-3600C18Q-128GTZR

## Motherboard
[LINK](https://www.newegg.com/msi-mag-b550m-mortar-max-wifi/p/N82E16813144636?Item=N82E16813144636) MSI MAG B550M MORTAR MAX WIFI DDR4 AM4 AMD B550 SATA 6Gb/s Micro-ATX Wi-Fi 6E 2.5Gbps LAN M.2 (Key-E) Motherboards - AMD

## Power Supply
[LINK](https://www.newegg.com/deepcool-r-pq850m-fa0b-us-850-w/p/N82E16817328036?Item=N82E16817328036) Deepcool PQ850M R-PQ850M-FA0B-US 850 W ATX12V V2.4 80 PLUS GOLD Certified Full Modular Active PFC Power Supply

## Install Ubuntu 22.4.05
No minimal and no third party driver installation. I used secure boot and had no issues so far, does not mean you won't so disable secure boot if it presents an issue.

I installed the following packages as a part of the server variant of ubuntu:
    * `g++`
    * `kde-desktop-plasma`
    * `linux-generic-hw-22.04`
    * `xrdp`

These let me connect to my server so perform remote work. For the sake of this repo's instruction length it is assumed you know how to work with these packages.

## xpu-smi
xpu-smi is currently only functioning for Intel Data Center GPus.

## GPU Monitoring Software
The best gpu monitor app on Linux so far is intel-gpu-tools intel_gpu_top [repo](https://github.com/tiagovignatti/intel-gpu-tools/blob/master/tools/intel_gpu_top.c) [tutorial](http://www.oldcai.com/ai/intel-gpu-tools/)
```bash
sudo apt-get install intel-gpu-tools
```

Use the application:
```bash
sudo intel_gpu_top
```

## Other monitoring gimmicks
The Sparkle editions allow for the passive monitoring of temperature via the RGB color spectrum.

![sparkle_temp_graph](supporting_graphics/sparkle_temp_graph.png)

## Resizeable BAR
If you have an Arc GPU, one of the first things you should do is enable Resizable BAR. Resizable BAR allows a computer’s processor to access the graphics card’s entire memory instead of in small chunks. The Arc GPUs currently require this feature to perform as intended. You can enable the feature in your motherboard’s BIOS.

Here are links on how to do this for some of the popular motherboard manufacturers:

  * [ASRock](https://www.asrock.com/support/faq.asp?id=498)
  * [Asus](https://www.asus.com/support/FAQ/1046107/)
  * [EVGA](https://www.evga.com/support/faq/FAQdetails.aspx?faqid=59772)
  * [Gigabyte](https://www.gigabyte.com/WebPage/785/NVIDIA_resizable_bar.html)
  * [MSI](https://www.msi.com/blog/unlock-system-performance-to-extreme-resizable-bar)

Verify Resizable BAR:
```bash
lspci -v |grep -A8 VGA
```

Result:
```bash
2d:00.0 VGA compatible controller: Intel Corporation Device 56a0 (rev 08) (prog-if 00 [VGA controller])
	Subsystem: Device 172f:3937
	Flags: bus master, fast devsel, latency 0, IRQ 114, IOMMU group 19
	Memory at fb000000 (64-bit, non-prefetchable) [size=16M]
	Memory at 7800000000 (64-bit, prefetchable) [size=16G]
	Expansion ROM at fc000000 [disabled] [size=2M]
	Capabilities: <access denied>
	Kernel driver in use: i915
	Kernel modules: i915
```

## Operating System:
Intel’s documentation recommends Ubuntu 22.04 LTS or newer. If you already have Ubuntu 22.04 LTS installed, ensure it’s fully updated.

The following is the OS I used.
```bash
cat /etc/os-release
```

```bash
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## Drivers and other required software

1. The Intel® Extension for Pytorch and TensorFlow requires a specific set of drivers for native Linux. Add the repositories.intel.com/graphics package repository to your Ubuntu installation:
```bash
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | 
sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt-get update
```

2. Add the Intel® oneAPI library repositories to your Ubuntu installation:
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
```

3. Open the `tensorflow-env/installed_packages_version.txt` and scroll down to the install line for apt. Note that there is a script to grab the working packages needed for OneAPI in case you dare venture into other older versions and want to record your results. I think the same logic if applied correctly will help with other projects as well.

4. Set the oneAPI Environment Variables
You will need to run the following command to activate the oneAPI environment variables after starting a new shell:

5. Set up the oneAPI environment variables for the current shell session
```bash
source /opt/intel/oneapi/setvars.sh
```
Alternatively, you can run the following command to add it to the .bashrc file:
```bash
echo 'source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1' >> ~/.bashrc
````

## Mamba Package Manager
We’ll use the [Mamba](https://mamba.readthedocs.io/en/latest/) package manager to create the Python environment. You can learn more about it in my [getting started](https://christianjmills.com/posts/mamba-getting-started-tutorial-windows/) tutorial.

The following bash commands will download the latest release, install it, and relaunch the current bash shell to apply the relevant changes:
```bash
# Download the latest Miniforge3 installer for the current OS and architecture
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Run the Miniforge3 installer silently (-b flag for batch mode)
bash Miniforge3-$(uname)-$(uname -m).sh -b

# Initialize mamba for shell usage
~/miniforge3/bin/mamba init

# Restart the shell to apply changes
bash
```

**Configuring a Miniforge Environment as a Python Interpreter in PyCharm**

To set up a Miniforge environment as your Python interpreter in PyCharm, follow these steps:

1. **Open Project Settings**:
   - In PyCharm, navigate to `File` > `Settings` (or `PyCharm` > `Preferences` on macOS).
   - In the left-hand pane, select `Project: [Your Project Name]` > `Python Interpreter`.

2. **Add a New Interpreter**:
   - Click the gear icon next to the interpreter dropdown and select `Add...`.
   - In the `Add Python Interpreter` dialog, choose `Existing environment`.

3. **Select the Python Executable**:
   - Click the `...` (browse) button next to the `Interpreter` field.
   - Navigate to your Miniforge environment's `bin` directory, typically located at `~/miniforge/envs/your-env/bin/`.
   - Select the `python3` executable (e.g., `python3.8` or `python3.9`).

4. **Apply the Configuration**:
   - Click `OK` to confirm your selection.
   - Back in the `Settings` window, ensure the new interpreter is selected, then click `Apply` and `OK` to apply the changes.

By completing these steps, PyCharm will utilize the specified Miniforge environment as the project's Python interpreter, ensuring consistency with your development setup. For more detailed information, refer to PyCharm's official documentation on configuring a Python interpreter: [Configure a Python interpreter | PyCharm Documentation](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)

Export your environment without build hashes and prefixes with the following command:
```bash
mamba env export --no-builds | grep -v "^prefix:" > environment.yml
```

## Pycharm Environment
1. In order to run the tensorflow script in pycharm you need to get the environment variables loaded into a run configuration.

2. In Pycharm you can do the following to export the environment variables from oneapi to use for tensorflow:
```bash
printenv | grep oneapi | paste -sd ';' -
```

Result:
```bash
TBBROOT=/opt/intel/oneapi/tbb/2022.0/env/..;ONEAPI_ROOT=/opt/intel/oneapi;PKG_CONFIG_PATH=/opt/intel/oneapi/tbb/2022.0/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.14/lib/pkgconfig:/opt/intel/oneapi/mkl/2025.0/lib/pkgconfig:/opt/intel/oneapi/ippcp/2025.0/lib/pkgconfig:/opt/intel/oneapi/dpl/2022.7/lib/pkgconfig:/opt/intel/oneapi/dnnl/2025.0/lib/pkgconfig:/opt/intel/oneapi/dal/2025.0/lib/pkgconfig:/opt/intel/oneapi/compiler/2025.0/lib/pkgconfig:/opt/intel/oneapi/ccl/2021.14/lib/pkgconfig/;CCL_ROOT=/opt/intel/oneapi/ccl/2021.14;I_MPI_ROOT=/opt/intel/oneapi/mpi/2021.14;FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.14/opt/mpi/libfabric/lib/prov:/usr/lib/x86_64-linux-gnu/libfabric;DNNLROOT=/opt/intel/oneapi/dnnl/2025.0;DIAGUTIL_PATH=/opt/intel/oneapi/dpcpp-ct/2025.0/etc/dpct/sys_check/sys_check.sh:/opt/intel/oneapi/compiler/2025.0/etc/compiler/sys_check/sys_check.sh;DPL_ROOT=/opt/intel/oneapi/dpl/2022.7;MANPATH=/opt/intel/oneapi/mpi/2021.14/share/man:/opt/intel/oneapi/debugger/2025.0/share/man:/opt/intel/oneapi/compiler/2025.0/share/man:;TCM_ROOT=/opt/intel/oneapi/tcm/1.2;GDB_INFO=/opt/intel/oneapi/debugger/2025.0/share/info/;CMAKE_PREFIX_PATH=/opt/intel/oneapi/tbb/2022.0/env/..:/opt/intel/oneapi/mkl/2025.0/lib/cmake:/opt/intel/oneapi/ipp/2022.0/lib/cmake/ipp:/opt/intel/oneapi/dpl/2022.7/lib/cmake/oneDPL:/opt/intel/oneapi/dnnl/2025.0/lib/cmake:/opt/intel/oneapi/dal/2025.0:/opt/intel/oneapi/compiler/2025.0;CMPLR_ROOT=/opt/intel/oneapi/compiler/2025.0;INFOPATH=/opt/intel/oneapi/debugger/2025.0/share/info;IPPROOT=/opt/intel/oneapi/ipp/2022.0;DALROOT=/opt/intel/oneapi/dal/2025.0;UMF_ROOT=/opt/intel/oneapi/umf/0.9;LIBRARY_PATH=/opt/intel/oneapi/tcm/1.2/lib:/opt/intel/oneapi/umf/0.9/lib:/opt/intel/oneapi/tbb/2022.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.14/lib:/opt/intel/oneapi/mkl/2025.0/lib:/opt/intel/oneapi/ippcp/2025.0/lib/:/opt/intel/oneapi/ipp/2022.0/lib:/opt/intel/oneapi/dnnl/2025.0/lib:/opt/intel/oneapi/dal/2025.0/lib:/opt/intel/oneapi/compiler/2025.0/lib:/opt/intel/oneapi/ccl/2021.14/lib/;IPPCRYPTOROOT=/opt/intel/oneapi/ippcp/2025.0;OCL_ICD_FILENAMES=/opt/intel/oneapi/compiler/2025.0/lib/libintelocl.so;CLASSPATH=/opt/intel/oneapi/mpi/2021.14/share/java/mpi.jar;LD_LIBRARY_PATH=/opt/intel/oneapi/tcm/1.2/lib:/opt/intel/oneapi/umf/0.9/lib:/opt/intel/oneapi/tbb/2022.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.14/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.14/lib:/opt/intel/oneapi/mkl/2025.0/lib:/opt/intel/oneapi/ippcp/2025.0/lib/:/opt/intel/oneapi/ipp/2022.0/lib:/opt/intel/oneapi/dnnl/2025.0/lib:/opt/intel/oneapi/debugger/2025.0/opt/debugger/lib:/opt/intel/oneapi/dal/2025.0/lib:/opt/intel/oneapi/compiler/2025.0/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.0/lib:/opt/intel/oneapi/ccl/2021.14/lib/;MKLROOT=/opt/intel/oneapi/mkl/2025.0;NLSPATH=/opt/intel/oneapi/compiler/2025.0/lib/compiler/locale/%l_%t/%N;PATH=/opt/intel/oneapi/mpi/2021.14/bin:/opt/intel/oneapi/mkl/2025.0/bin:/opt/intel/oneapi/dpcpp-ct/2025.0/bin:/opt/intel/oneapi/dev-utilities/2025.0/bin:/opt/intel/oneapi/debugger/2025.0/opt/debugger/bin:/opt/intel/oneapi/compiler/2025.0/bin:/home/flaniganp/miniforge3/envs/tensorflow-arc/bin:/home/flaniganp/miniforge3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin;INTEL_PYTHONHOME=/opt/intel/oneapi/debugger/2025.0/opt/debugger;CPATH=/opt/intel/oneapi/umf/0.9/include:/opt/intel/oneapi/tbb/2022.0/env/../include:/opt/intel/oneapi/mpi/2021.14/include:/opt/intel/oneapi/mkl/2025.0/include:/opt/intel/oneapi/ippcp/2025.0/include:/opt/intel/oneapi/ipp/2022.0/include:/opt/intel/oneapi/dpl/2022.7/include:/opt/intel/oneapi/dpcpp-ct/2025.0/include:/opt/intel/oneapi/dnnl/2025.0/include:/opt/intel/oneapi/dev-utilities/2025.0/include:/opt/intel/oneapi/dal/2025.0/include:/opt/intel/oneapi/ccl/2021.14/include
```

3. Create a new runtime configuration.

![env_setup_1](supporting_graphics/pycharm/env_setup_1.png)

4. Select the "+".

![env_setup_2](supporting_graphics/pycharm/env_setup_2.png)

5. Select Python.

![env_setup_3](supporting_graphics/pycharm/env_setup_3.png)

6. Fill in Name, script and place a semicolon after PYTHONBUFFERED=1. After the semicolon place the semicolon delimited env from step 8.

![env_setup_4](supporting_graphics/pycharm/env_setup_4.png)

7. When you script runs you will know when the GPU is utilized when the following pops up:
```bash
Intel Extension for Tensorflow* GPU backend is loaded.
```

### Intel Extension for Pytorch
1. Create a Python Environment:

Next, we’ll create a Python environment and activate it. The current version of the extension supports Python 3.10, so we’ll use that.
```bash
mamba env create -f pytorch-env/pytorch-arc.yml
mamba activate pytorch-arc
```

2. Sanity check.
```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

Result (Ignore the warning for raytracing unless you want to proceed in a different direction than this tutorial), I installed the `intel-level-zero-gpu-raytracing` but still encountered this issue:
```bash
[W1231 20:41:53.115852190 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
    registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
2.5.1+cxx11.abi
2.5.10+xpu
[0]: _XpuDeviceProperties(name='Intel(R) Arc(TM) A770 Graphics', platform_name='Intel(R) oneAPI Unified Runtime over Level-Zero', type='gpu', driver_version='1.3.27642', total_memory=15473MB, max_compute_units=512, gpu_eu_count=512, gpu_subslice_count=32, max_work_group_size=1024, max_num_sub_groups=128, sub_group_sizes=[8 16 32], has_fp16=1, has_fp64=0, has_atomic64=1)
```

3. Install additional dependencies:

Here is the jupyter notebook environment that is separate from the standalone pytorch. You can learn about these dependencies ([here](https://christianjmills.com/posts/pytorch-train-image-classifier-timm-hf-tutorial/#installing-additional-libraries)).
```bash
mamba env create -f pytorch-env/jupyter-notebook-environment.yml
mamba activate pytorch-arc-jupyter-notebook-env
```

#### Modify PyTorch Code
It’s finally time to train a model. The Jupyter Notebooks with the original and modified training code are available on GitHub at the links below.

[pytorch-timm-image-classifier-training.ipynb](https://github.com/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/blob/main/notebooks/pytorch-timm-image-classifier-training.ipynb)
[intel-arc-pytorch-timm-image-classifier-training.ipynb](https://github.com/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/blob/main/notebooks/intel-arc-pytorch-timm-image-classifier-training.ipynb)

You can also download the notebooks to the current directory by running the following commands:
```bash
wget https://raw.githubusercontent.com/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/main/notebooks/pytorch-timm-image-classifier-training.ipynb \
wget https://raw.githubusercontent.com/cj-mills/pytorch-timm-gesture-recognition-tutorial-code/main/notebooks/intel-arc-pytorch-timm-image-classifier-training.ipynb
```

4. Once downloaded, run the following command to launch the Jupyter Notebook Environment:
```bash
jupyter notebook
```

# The rest of the steps are referencing the *modified* jupyter notebook.

5. Set Environment Variables
First, we need to set the following environment variables:
```bash
import os
os.environ['OCL_ICD_VENDORS'] = '/etc/OpenCL/vendors'
os.environ['CCL_ROOT'] = os.environ.get('CONDA_PREFIX', '')
```

6. . We import Intel’s PyTorch extension with the following code:
```bash
import torch
import intel_extension_for_pytorch as ipex

print(f'PyTorch Version: {torch.__version__}')
print(f'Intel PyTorch Extension Version: {ipex.__version__}')
```

Note that we need to import PyTorch before the extension. Also, if you get the following user warning about transformers, don’t worry. It’s normal.

7. We don’t want to re-import torch after the extension, so we’ll remove that line from the Import PyTorch dependencies section.
```bash
# Import PyTorch dependencies
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
```

8. Verify Arc GPU Availability
```bash
import pandas as pd

def get_public_properties(obj):
    """
    Extract all public properties from an object.

    Args:
    obj: The object to extract properties from.

    Returns:
    dict: A dictionary containing the object's public properties and their values.
    """
    return {
        prop: getattr(obj, prop)
        for prop in dir(obj)
        if not prop.startswith("__") and not callable(getattr(obj, prop))
    }

# Get the number of available XPU devices
xpu_device_count = torch.xpu.device_count()

# Create a list of dictionaries containing properties for each XPU device
dict_properties_list = [
    get_public_properties(torch.xpu.get_device_properties(i))
    for i in range(xpu_device_count)
]

# Convert the list of dictionaries to a pandas DataFrame for easy viewing
pd.DataFrame(dict_properties_list)
```

9. Next, we’ll manually set the device name to xpu.
```bash
device = 'xpu'
dtype = torch.float32
device, dtype
```

10. Optimize the model and optimizer Objects
```bash
# Learning rate for the model
lr = 1e-3

# Number of training epochs
epochs = 3

# AdamW optimizer; includes weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

# Optimize the model and optimizer objects
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

# Learning rate scheduler; adjusts the learning rate during training
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                   max_lr=lr,
                                                   total_steps=epochs*len(train_dataloader))

# Performance metric: Multiclass Accuracy
metric = MulticlassAccuracy()
```

11. Train the Model
That’s it for the required changes to the training code. We can now run the train_loop function.

12. Update the Inference Code
Since we cast the model to float16, we must ensure input data use the same type. We can update the inference code using the auto-cast context manager as shown below:
```bash
# Make a prediction with the model
with torch.no_grad(), autocast(torch.device(device).type):
    pred = model(img_tensor)
```

### Local LLM Inference with `IPEX-LLM`
To close out this tutorial, we will cover how to perform local LLM inference using Intel’s ipex-llm library. This library allows us to run many popular LLMs in INT4 precision on our Arc GPU.

1. Set Environment Variables in a *new* terminal without the set vars from either the Pytorch or Tensorflow exercise in this README:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/intel/oneapi/lib
export PYTHONUSERBASE=~/intel/oneapi
export PATH=~/intel/oneapi/bin:$PATH
```

These commands update your environment variables to include the oneAPI library and binary paths.

2. Create and Activate a Conda Environment:
```bash
mamba create --name ipex-llm-env python=3.10 -y
mamba activate ipex-llm-env
```

This creates a new Conda environment named ipex-llm-env with Python 3.10 and activates it.

3. Install Intel Extension for PyTorch:
```bash
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --user
```

This installs the pre-release version of Intel Extension for PyTorch with XPU support.

4. Install Additional Python Packages:
```bash
pip install jupyter transformers==4.43.1 trl --user
```

This installs Jupyter, a specific version of the Transformers library, and the trl package.

5. Install Intel OneAPI Libraries:
```bash
pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0 --user
```

This installs specific versions of Intel's oneAPI libraries.

Note: The --user flag installs the packages to the user’s site-packages directory. Ensure that the PYTHONUSERBASE environment variable is set appropriately to include these packages.

Important: When setting environment variables like LD_LIBRARY_PATH, be cautious as it can affect system-wide library loading. It's recommended to set such variables within the scope of your Conda environment to avoid potential conflicts.

2. We can launch a new Jupyter Notebook environment once the dependencies finish installing.
```bash
jupyter notebook
```

3. Enter acces token from Hugging Face (read permission for your requested model `meta-llama/Meta-Llama-3.1-8B-Instruct`):
```bash
from getpass import getpass

# Get from you hugging face account
api_token = getpass('Enter your API token: ')
```

4. Set Environment Variables
With our environment set up, we can dive into the code. First, we need to set the following environment variables:
```bash
import os
os.environ['OCL_ICD_VENDORS'] = '/etc/OpenCL/vendors'
os.environ['CCL_ROOT'] = os.environ.get('CONDA_PREFIX', '')
os.environ['USE_XETLA'] = 'OFF'
os.environ['SYCL_CACHE_PERSISTENT'] = '1'
os.environ['SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS'] = '1'
```

5. Import the Required Dependencies
Next, we will import the necessary Python packages into our Jupyter Notebook.
```bash
import torch
import time
import argparse

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from IPython.display import Markdown
```

We can use the Markdown class from IPython to render Markdown output from the model inside the notebook.

6. Define a Function to Prepare the Prompt
We can use the following function from this example script to prepare prompts for the LLama 3.1 model:

```bash
def get_prompt(user_input: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
    """
    Generate a formatted prompt for a LLaMA 3.1 chatbot conversation.

    This function takes the user's input, chat history, and system prompt,
    and combines them into a single formatted string for use in a LLaMA 3.1 chatbot system.

    Parameters:
    user_input (str): The current input from the user.
    chat_history (list[tuple[str, str]]): A list of tuples containing previous
                                          (user_input, assistant_response) pairs.
    system_prompt (str): Initial instructions or context for the LLaMA 3.1 chatbot.

    Returns:
    str: A formatted string containing the entire conversation history and current input.
    """

    # Start the prompt with a special token
    prompt_texts = ['<|begin_of_text|>']

    # Add system prompt if it's not empty
    if system_prompt != '':
        prompt_texts.append(f'<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>')

    # Add each pair of user input and assistant response from the chat history
    for history_input, history_response in chat_history:
        prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{history_input.strip()}<|eot_id|>')
        prompt_texts.append(f'<|start_header_id|>assistant<|end_header_id|>\n\n{history_response.strip()}<|eot_id|>')

    # Add the current user input and prepare for assistant's response
    prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')

    # Join all parts of the prompt into a single string
    return ''.join(prompt_texts)
```

7. Load the Model in INT4
Next, we can load the LLaMA 3.1 8B Instruct model in 4-bit precision.

*Important*
You will need to accept Meta’s license agreement through the model’s HuggingFace Hub page to access the model:
```bash
meta-llama/Meta-Llama-3.1-8B-Instruct
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load model in 4 bit, which converts the relevant layers in the model into INT4 format
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             optimize_model=True,
                                             trust_remote_code=True,
                                             use_cache=True)
model = model.half().to('xpu')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

Define Inference Parameters
Before running the model, we must define our prompts and the maximum number of tokens the model should generate.

DEFAULT_SYSTEM_PROMPT = """\
"""

prompt_str = "Provide a clear, concise, and intuitive description of AI for beginners."
max_new_tokens = 512
```

8. Perform Inference
Finally, we can run the model.

```bash
# Disable gradient computation for inference
with torch.inference_mode():
    # Generate the input prompt using a custom function
    prompt = get_prompt(prompt_str, [], system_prompt=DEFAULT_SYSTEM_PROMPT)

    # Encode the prompt into token IDs and move to the XPU (Intel's GPU)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')

    # Perform a warmup run to optimize the model
    # This run's output is discarded
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # Start the actual inference
    st = time.time()  # Record the start time

    # Generate the output using the language model
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # Ensure all XPU operations are completed before recording end time
    torch.xpu.synchronize()

    end = time.time()  # Record the end time

    # Move the output back to CPU for further processing
    output = output.cpu()

    # Decode the output tokens into a human-readable string
    # skip_special_tokens=False means we keep all special tokens in the output
    output_str = tokenizer.decode(output[0], skip_special_tokens=False)

    # Print the inference time
    print(f'Inference time: {end-st:.2f} seconds')

    # Print the original prompt
    print('-'*20, 'Prompt', '-'*20)
    print(prompt)
```

9. Print the model's response
```bash
print('-'*20, 'Response (skip_special_tokens=False)', '-'*20)
```

10. Extract the actual response from the output string
This assumes the response is between the last '<|end_header_id|>\n\n' and '<|eot_id|>'
```bash
response = output_str.split('<|end_header_id|>\n\n')[-1].split('<|eot_id|>')[0]


# Display the response using Markdown formatting
Markdown(response)

Inference time: 51.17 seconds
-------------------- Prompt --------------------
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Provide a clear, concise, and intuitive description of AI for beginners.<|eot_id|><|start_header_id|>assistant<|end_header_id|>


-------------------- Response (skip_special_tokens=False) --------------------
```

Model Response:
```bash
Here’s a clear, concise, and intuitive description of AI for beginners:

What is AI?

Artificial Intelligence (AI) is a type of computer technology that allows machines to think and learn like humans. It’s a way to create intelligent systems that can perform tasks that typically require human intelligence, such as:

Recognizing images and objects
Understanding natural language
Making decisions
Solving problems
Learning from data
How does AI work?

AI uses a combination of algorithms, data, and computer power to mimic human thought processes. Here’s a simplified overview:

Data collection: Large amounts of data are gathered from various sources, such as images, text, or sensor readings.
Pattern recognition: AI algorithms analyze the data to identify patterns, relationships, and trends.
Machine learning: The AI system uses this analysis to make predictions, classify data, or make decisions.
Improvement: The AI system learns from its experiences and adjusts its performance over time.
Types of AI

There are several types of AI, including:

Narrow or Weak AI: Designed to perform a specific task, like facial recognition or language translation.
Strong or General AI: A hypothetical AI that can perform any intellectual task that humans can.
Superintelligence: A hypothetical AI that surpasses human intelligence in all domains.
Key benefits of AI

AI has many benefits, including:

Improved efficiency: AI can automate repetitive tasks, freeing up human time for more complex and creative work.
Enhanced decision-making: AI can analyze vast amounts of data to make informed decisions.
Personalization: AI can tailor experiences to individual preferences and needs.
Key concepts

Some key concepts to keep in mind:

Machine learning: A type of AI that improves over time based on data experience.
Deep learning: A type of machine learning that uses neural networks to analyze complex data.
Natural language processing: A type of AI that enables computers to understand and generate human-like language.
That’s a basic introduction to AI for beginners! I hope this helps you understand the basics of this exciting and rapidly evolving field.
```

### Intel Extension for Tensorflow
The following instructions were gleaned from [here](https://intel.github.io/intel-extension-for-tensorflow/latest/get_started.html) and the Christian Mills blog from Pytorch. 

1. Create and activate environment for tensorflow-arc.
```bash
mamba env create -f tensorflow-env/tensorflow-environment.yml
mamba activate tensorflow-arc
```

2. Check environment:
```bash
export path_to_site_packages=`python -c "import site; print(site.getsitepackages()[0])"`
python ${path_to_site_packages}/intel_extension_for_tensorflow/tools/python/env_check.py
```

Result:
```bash
Check Environment for Intel(R) Extension for TensorFlow*...

__file__:     /home/flaniganp/miniforge3/envs/tensorflow-arc/lib/python3.10/site-packages/intel_extension_for_tensorflow/tools/python/env_check.py
Check Python
         Python 3.10.16 is Supported.
Check Python Passed

Check OS
        OS ubuntu:22.04 is Supported
Check OS Passed

Check Tensorflow
2025-01-01 00:37:19.216775: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-01-01 00:37:19.239756: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-01-01 00:37:19.239783: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-01-01 00:37:19.240687: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-01-01 00:37:19.244902: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-01-01 00:37:19.245042: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-01 00:37:19.684969: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2025-01-01 00:37:20.245827: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /tensorflow/core/bfc_allocator_delay. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:37:20.245948: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /xla/service/gpu/compiled_programs_count. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:37:20.246684: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_executions. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:37:20.246696: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_execution_time_usecs. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:37:20.360327: I itex/core/wrapper/itex_gpu_wrapper.cc:38] Intel Extension for Tensorflow* GPU backend is loaded.
2025-01-01 00:37:20.360619: I external/local_xla/xla/pjrt/pjrt_api.cc:67] PJRT_Api is set for device type xpu
2025-01-01 00:37:20.360630: I external/local_xla/xla/pjrt/pjrt_api.cc:72] PJRT plugin for XPU has PJRT API version 0.33. The framework PJRT API version is 0.34.
2025-01-01 00:37:20.374992: I external/intel_xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:134] Selected platform: Intel(R) oneAPI Unified Runtime over Level-Zero
2025-01-01 00:37:20.375177: I external/intel_xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:159] number of sub-devices is zero, expose root device.
2025-01-01 00:37:20.375840: I external/xla/xla/service/service.cc:168] XLA service 0x61779c3095a0 initialized for platform SYCL (this does not guarantee that XLA will be used). Devices:
2025-01-01 00:37:20.375849: I external/xla/xla/service/service.cc:176]   StreamExecutor device (0): Intel(R) Arc(TM) A770 Graphics, <undefined>
2025-01-01 00:37:20.376739: I itex/core/devices/gpu/itex_gpu_runtime.cc:130] Selected platform: Intel(R) oneAPI Unified Runtime over Level-Zero
2025-01-01 00:37:20.376918: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2025-01-01 00:37:20.377281: I external/intel_xla/xla/pjrt/se_xpu_pjrt_client.cc:97] Using BFC allocator.
2025-01-01 00:37:20.377295: I external/xla/xla/pjrt/gpu/gpu_helpers.cc:106] XLA backend allocating 14602718822 bytes on device 0 for BFCAllocator.
2025-01-01 00:37:20.378384: I external/local_xla/xla/pjrt/pjrt_c_api_client.cc:119] PjRtCApiClient created.
        Tensorflow 2.15.0 is installed.
Your Tensorflow version is too low, please upgrade to 2.15.1!
```

8. Verify the installation via python import.
```bash
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```

Result:

```bash
2025-01-01 00:36:29.154297: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-01-01 00:36:29.177656: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-01-01 00:36:29.177677: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-01-01 00:36:29.178594: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-01-01 00:36:29.183008: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-01-01 00:36:29.183145: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-01 00:36:29.624909: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2025-01-01 00:36:30.178012: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /tensorflow/core/bfc_allocator_delay. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:36:30.178128: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /xla/service/gpu/compiled_programs_count. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:36:30.178845: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_executions. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:36:30.178856: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_execution_time_usecs. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 00:36:30.291245: I itex/core/wrapper/itex_gpu_wrapper.cc:38] Intel Extension for Tensorflow* GPU backend is loaded.
2025-01-01 00:36:30.291527: I external/local_xla/xla/pjrt/pjrt_api.cc:67] PJRT_Api is set for device type xpu
2025-01-01 00:36:30.291538: I external/local_xla/xla/pjrt/pjrt_api.cc:72] PJRT plugin for XPU has PJRT API version 0.33. The framework PJRT API version is 0.34.
2025-01-01 00:36:30.305963: I external/intel_xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:134] Selected platform: Intel(R) oneAPI Unified Runtime over Level-Zero
2025-01-01 00:36:30.306147: I external/intel_xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:159] number of sub-devices is zero, expose root device.
2025-01-01 00:36:30.306751: I external/xla/xla/service/service.cc:168] XLA service 0x64fc7afa7870 initialized for platform SYCL (this does not guarantee that XLA will be used). Devices:
2025-01-01 00:36:30.306760: I external/xla/xla/service/service.cc:176]   StreamExecutor device (0): Intel(R) Arc(TM) A770 Graphics, <undefined>
2025-01-01 00:36:30.307606: I itex/core/devices/gpu/itex_gpu_runtime.cc:130] Selected platform: Intel(R) oneAPI Unified Runtime over Level-Zero
2025-01-01 00:36:30.307787: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2025-01-01 00:36:30.308105: I external/intel_xla/xla/pjrt/se_xpu_pjrt_client.cc:97] Using BFC allocator.
2025-01-01 00:36:30.308120: I external/xla/xla/pjrt/gpu/gpu_helpers.cc:106] XLA backend allocating 14602718822 bytes on device 0 for BFCAllocator.
2025-01-01 00:36:30.309094: I external/local_xla/xla/pjrt/pjrt_c_api_client.cc:119] PjRtCApiClient created.
2.15.0.2
```

16. Run the example tensorflow script which was taken from [Daniel Bourke's Tensorflow](https://dev.mrdbourke.com/tensorflow-deep-learning/) class's repo with some changes to print the tensorflow version.
```bash
python tensorflow-env/tensorflow-hello-world.py
```

Result:
```bash
2025-01-01 01:10:21.926246: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-01-01 01:10:21.949076: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-01-01 01:10:21.949096: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-01-01 01:10:21.949998: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-01-01 01:10:21.954119: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2025-01-01 01:10:21.954256: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-01 01:10:22.409722: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2025-01-01 01:10:22.990277: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /tensorflow/core/bfc_allocator_delay. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 01:10:22.990382: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /xla/service/gpu/compiled_programs_count. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 01:10:22.991106: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_executions. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 01:10:22.991116: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_execution_time_usecs. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-01-01 01:10:23.109972: I itex/core/wrapper/itex_gpu_wrapper.cc:38] Intel Extension for Tensorflow* GPU backend is loaded.
2025-01-01 01:10:23.110243: I external/local_xla/xla/pjrt/pjrt_api.cc:67] PJRT_Api is set for device type xpu
2025-01-01 01:10:23.110254: I external/local_xla/xla/pjrt/pjrt_api.cc:72] PJRT plugin for XPU has PJRT API version 0.33. The framework PJRT API version is 0.34.
2025-01-01 01:10:23.124704: I external/intel_xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:134] Selected platform: Intel(R) oneAPI Unified Runtime over Level-Zero
2025-01-01 01:10:23.124888: I external/intel_xla/xla/stream_executor/sycl/sycl_gpu_runtime.cc:159] number of sub-devices is zero, expose root device.
2025-01-01 01:10:23.125821: I external/xla/xla/service/service.cc:168] XLA service 0x5d0aa8be8410 initialized for platform SYCL (this does not guarantee that XLA will be used). Devices:
2025-01-01 01:10:23.125832: I external/xla/xla/service/service.cc:176]   StreamExecutor device (0): Intel(R) Arc(TM) A770 Graphics, <undefined>
2025-01-01 01:10:23.126678: I itex/core/devices/gpu/itex_gpu_runtime.cc:130] Selected platform: Intel(R) oneAPI Unified Runtime over Level-Zero
2025-01-01 01:10:23.126859: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2025-01-01 01:10:23.127235: I external/intel_xla/xla/pjrt/se_xpu_pjrt_client.cc:97] Using BFC allocator.
2025-01-01 01:10:23.127249: I external/xla/xla/pjrt/gpu/gpu_helpers.cc:106] XLA backend allocating 14602718822 bytes on device 0 for BFCAllocator.
2025-01-01 01:10:23.128320: I external/local_xla/xla/pjrt/pjrt_c_api_client.cc:119] PjRtCApiClient created.
TensorFlow version: 2.15.0
Intel Extension Version: 2.15.0.2
CUDA version: 12.2
cuDNN version: 8
Backend utilized: GPU
Original Label: 9
One-Hot Encoded Label: [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
Label: 4
Class Name: Coat
Image Shape: (28, 28)
Print the train image shape (60000, 28, 28).
Print the train labels shape (60000,).
2025-01-01 01:10:25.257602: I tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_factory.cc:118] Created 1 TensorFlow NextPluggableDevices. Physical device type: XPU
Training model...
Epoch 1/10
2025-01-01 01:10:26.017015: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type XPU is enabled.
1875/1875 [==============================] - 12s 6ms/step - loss: 0.4619 - accuracy: 0.8334 - val_loss: 0.3786 - val_accuracy: 0.8694
Epoch 2/10
1875/1875 [==============================] - 11s 6ms/step - loss: 0.3317 - accuracy: 0.8783 - val_loss: 0.3374 - val_accuracy: 0.8813
Epoch 3/10
1875/1875 [==============================] - 14s 8ms/step - loss: 0.3061 - accuracy: 0.8896 - val_loss: 0.3449 - val_accuracy: 0.8748
Epoch 4/10
1875/1875 [==============================] - 13s 7ms/step - loss: 0.2869 - accuracy: 0.8971 - val_loss: 0.3082 - val_accuracy: 0.8794
Epoch 5/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.2708 - accuracy: 0.9021 - val_loss: 0.2971 - val_accuracy: 0.8897
Epoch 6/10
1875/1875 [==============================] - 14s 8ms/step - loss: 0.2569 - accuracy: 0.9052 - val_loss: 0.2995 - val_accuracy: 0.8922
Epoch 7/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.2480 - accuracy: 0.9076 - val_loss: 0.2910 - val_accuracy: 0.8943
Epoch 8/10
1875/1875 [==============================] - 14s 7ms/step - loss: 0.2417 - accuracy: 0.9111 - val_loss: 0.2957 - val_accuracy: 0.8977
Epoch 9/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.2297 - accuracy: 0.9135 - val_loss: 0.2899 - val_accuracy: 0.8982
Epoch 10/10
1875/1875 [==============================] - 14s 8ms/step - loss: 0.2195 - accuracy: 0.9173 - val_loss: 0.3093 - val_accuracy: 0.8922
Evaluating model...
313/313 [==============================] - 1s 4ms/step - loss: 0.3218 - accuracy: 0.8922
/home/flaniganp/miniforge3/envs/tensorflow-arc/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

Process finished with exit code 0
```

17. When you script runs you will know when the GPU is utilized when the following pops up:
```bash
Intel Extension for Tensorflow* GPU backend is loaded.
```

# Conclusion
In this tutorial, we set up Intel’s PyTorch extension on Ubuntu and trained an image classification model using an Arc GPU. The exact setup steps may change with new versions, so check the documentation for the latest version to see if there are any changes. I’ll try to keep this tutorial updated with any significant changes to the process and to keep in line with the original poster's information.

# Tips
## Intel DKMS
1. If at any time this package, [intel-i915-dkms](https://www.reddit.com/r/linuxquestions/comments/11afpdz/inteli915dkms_nightmare/?rdt=47052), gets in the way of installing or upgrading packages. Get rid of it by.
```bash
sudo apt purge intel-i915-dkms
```

2. Then remove the package and any other ones not being used.
```bash
sudo apt autoremove
```

3. Kernel Error. Currently, I have this pop up in my tensorflow runs but I don't know what to do about it or what it is affecting...
```bash
Instruction / Operand / Region Errors:

/-------------------------------------------!!!KERNEL HEADER ERRORS FOUND!!!-------------------------------------------\
Error in CISA routine with name: igc_check
                  Error Message: Input V38 = [256, 260) intersects with V37 = [256, 260)
\----------------------------------------------------------------------------------------------------------------------/
```