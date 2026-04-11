---
marp: true
theme: gaia
_class: lead
paginate: true
style: |
  footer img {
    height: 50px;
  }
  .speaker-row {
    display: flex;
    justify-content: center;   /* center horizontally */
    gap: 32px;                 /* space between images */
    margin-bottom: 6px;
  }
  .speaker-row img {
    width: 150px;
  }
  .speaker-caption {
    text-align: center;
    font-size: 0.9em;
    color: #555;
  }
  section.lead {
    text-align: center;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
  }
  .small {
    font-size: 0.8em;
  }
---

# Speakers


<!-- _footer: ![EPICURE logo](Epicure.png) -->

<!-- _class: lead -->

# Understanding why your GPU-accelerated is slow using NVIDIA Nsight Systems
Apr 15, 2026 | 1:20 PM - 3:00 PM

![alt text](Scynergy.png)

___
# Presenters 

<!-- _class: lead -->

<div class="speaker-row">
  <img src="Marco.png" alt="Marco Magliulo">
  <img src="Tom.png" alt="Tom Walter">
</div>

<div class="speaker-caption">
  Marco Magliulo &nbsp;&nbsp;|&nbsp;&nbsp; Tom Walter
</div>


---

# Goal of this workshop

By the end, you should be able to:

- Profile your GPU jobs on Meluxina
- Interpret key trace metrics and timelines  
- Identify common GPU bottlenecks (IO, compute, memory, synchronization, communication)  
- Apply simple optimizations and validate improvements  

---

## Agenda 

- Connection to Meluxina via OpenOnDemand
- Introduction to NVIDIA NSight-Systems
- Hands-on: making a PyTorch Training faster
  - distributing it 
  - making it faster


---

<!-- _class: lead -->

# Connecting to Meluxina via OpenOnDemand 


---
# https://portal.lxp.lu/

![width:700px](image.png)

---
# Openning the Desktop app 

![width:700px](image-1.png)

---
# Choosing the appropriate job options 

![width:700px](image-2.png)

---
# Accessing the session

![width:700px](image-3.png)

---
# Openning the terminal app

![width:900px](image-4.png)



---

<!-- _class: lead -->

# Getting the code  


---
# Going to the project folder 


```bash
cd /project/home/p201259/workspaces/ 
mkdir -p $USER/
cd $USER/
```

---
# Cloning the repo

```bash
git clone https://github.com/LuxProvide/Scynergy2026-GPUApplicationProfiling
cd Scynergy2026-GPUApplicationProfiling/
```

---
##  Key Questions Answered via Profiling

- **CPU/IO Bottlenecks:** Is the GPU idle during data loading?
- **Data Movement:** Do H2D transfers dominate compute time?
- **Compute vs. Memory:** bandwidth-bound or compute-bound?
- **Multi-GPU Scaling:** Is there load imbalance across ranks?
- **Sync Stalls:** Are MPI/NCCL/Barriers causing idleness?
- **Launch Overhead:** Are kernels too small or frequent?
- **Occupancy:** Is register/SRAM usage limiting parallelism?

---

# Why Measuring/profiling GPU code?

- Wall‑clock runtime alone doesn’t explain *why* a job is slow
- GPU programming adds complexity:
  - Host ↔ device transfers
  - Kernel launches and occupancy
  - Memory hierarchy (global / shared / L2 / registers)

---

# NVIDIA Nsight tool family

- **Nsight Systems**
  - System‑wide timeline (CPU, GPU, MPI, I/O)
  - Good for: *“Where is the time going?”*
- **Nsight Compute**
  - Kernel‑level analysis and metrics
  - Good for: *“Why is this kernel so slow?”*

Today focus on **Nsight Systems** 

---

# Usual workflow

1. **Reproduce the problem** with a smaller test case
2. Run **Nsight Systems** on this smaller test case
3. Identify **top time consumers** in the timeline
4. Formulate hypotheses → apply changes → re‑profile
5. Repeat until performance is satisfactory 


---

# High-level workflow: Running Nsight on MeluXina 

Two main steps:
- Nsight-Systems produces a trace (`.nsys-rep` extension)
- We use the GUI of Nsight-Systems and its command line tools to analyze this trace 

---

# Our workflow for today 

For the ease of use, we are going to use OpenOnDemand.
We will then be able to:
- Open `Nsight-Systems` GUI to analyze traces already prepared for you,
- We will also use the rich `nsys` command line tools
- Modify the code, profile, adjust, start again

--- 

# First step: setting up the environment

Open a terminal in your OOD session and run:

```bash
cd /project/home/p201259/workspaces/$USER/Scynergy2026-GPUApplicationProfiling/
cd Scripts
source setup_environment.sh
```

--- 
# Second step: let's open a trace

```bash
module load Nsight-Systems
THE_TRACE=/mnt/tier2/project/p201259/materials/15April_GPUApp_Profiling/ProfilingTraces/single_gpu_base.nsys-rep
nsys-ui $THE_TRACE
```

--- 
# Second step: let's open a trace

![width:900px](<Screenshot 2026-04-11 at 13.04.13.png>)

___

# Let's have a closer look

![width:900px](image-5.png)


---

# Zoom on a part of the timeline

![width:900px](image-6.png)

---
# Filter and zoom in

![width:900px](image-7.png)


---
# Filter and zoom in

![width:900px](image-8.png)

---
# Zooming further

![width:900px](<Screenshot 2026-04-11 at 14.22.38.png>)


---
# Identifying the culprit 

![width:900px](image-10.png)

---
# Identifying the culprit 


![width:600px](image-9.png)

---

# First observations

From the screenshot alone:
✅ GPU is poorly utilized
✅ Memory usage is stable but low
⚠️ Almost everything is on default stream
⚠️ Limited concurrent execution
✅ CPU is active, not idle
⚠️ Long GPU gaps in between the training steps   

___
# Side note 

The analysis summary allows you to retrieve which command line you used to obtain the trace.
-> This can be very handy if you have a lot of traces 

![width:600px](image-11.png)

---
# Let's dig into the command to generate the trace

```bash
srun ${SRUN_OPTIONS} nsys-profile ${NSYS_OPTIONS} ${TORCHRUN_COMMAND}
```

---

```
NSYS_OPTIONS="--cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output=${output_file} \
    -t cuda,nvtx"
```

- **`--cuda-memory-usage`**: Tracks VRAM footprint 
- **`--capture-range=cudaProfilerApi`**: Only profiles the code between `start()` and `stop()` calls in the python code 
- **`--output`**: Defines the path for the `.nsys-rep` file.
- **`-t cuda`**: Traces GPU kernels, memory copies, and API calls.
- **`-t nvtx`**: Traces user-defined code annotations (e.g., "Epoch 1", "Optimizer").


---
### We only profile what we need (when possible)

Those 2 flags:

```bash
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
```
in conjunction with these functions:
```
import torch.cuda.profiler as profiler
profiler.start()
...
profiler.stop()
```

allow us to profile only what we need ! 

---


<!-- _class: lead -->

# Your turn to look at a trace 


---

# Foreword

run the following: 

```bash
module load Nsight-Systems
THE_TRACE=/mnt/tier2/project/p201259/materials/15April_GPUApp_Profiling/ProfilingTraces/multigpus_base.nsys-rep
nsys-ui $THE_TRACE
```

---
## Observations

![width:800px](image-12.png)

---
## Observations

- still a lot of gaps in the individual activity of the GPU in the distributed training
- single GPU: 217 sec for the epoch
- 4 GPUs: 179 sec

⚠️ 4x more GPU power but 18% improvement in runtime 

---
## What can kill the performance that much ? 

![width:800px](image-13.png)

---


---

# Workflow on MeluXina (summary)

1. Prepare a **representative but smaller** test case  
2. Request GPU node(s) via Slurm  
3. Run Nsight Systems for a **global view**  
4. Identify 1–3 **hot kernels**  
5. Run Nsight Compute targeted at those kernels  
6. Implement optimizations → re‑run benchmarks  
7. Once satisfied, scale back up to full production sizes

---

# Practical tips

- Start with **short profiling runs** to reduce overhead  
- Use **environment modules** to load correct CUDA/Nsight versions  
- Keep **profiling configs** (scripts, flags) in version control  
- Always compare **before vs after** with the same test case  
- Document:
  - What you changed
  - Which metrics improved
  - Any trade‑offs (e.g. memory vs speed)

---

# Hands‑on exercise (if time permits)

- We’ll take a simple GPU‑accelerated mini‑app  
- Steps:
  - Baseline run: measure runtime
  - Nsight Systems profile: inspect timeline
  - Nsight Compute: inspect top kernel
  - Apply 1–2 optimizations (e.g. block size, memory layout)
  - Re‑profile and discuss results

---

# What you can do after this workshop

- Apply Nsight to your **own applications** on MeluXina  
- Build a small **profiling checklist** for new codes  
- Share **profiling results** with colleagues to guide optimization  
- Reach out to support / performance teams with:
  - Profiles
  - Clear descriptions of bottlenecks

---

# Q & A

Questions, specific applications, or issues you’d like to discuss?

---

# Thank you

- Slides and example scripts:  
  - (Add link / repository here)
- Contact:  
  - Your email / group contact


---

<!-- _class: lead -->

# Back-up slides  

---

# Meluxina GPU node Hardware 

- CPU: 2× AMD 7452 EPYC ROME CPUs: 32 cores each
- GPUs:
    - 4× NVIDIA A100 GPUs on each node
      - 40 GB HBM2 each (the so-called VRAM)
    - NVLink between GPUs **of the same node**
- ~512 GB RAM 

---

# Meluxina GPU node Hardware 

- Storage / FS
    - Parallel filesystem (Lustre) for scratch/project storage
    - Local SSD for node‑local temporary data ~1.8 Tb 
- High‑speed HDR/InfiniBand (200 Gb/s ) between nodes
