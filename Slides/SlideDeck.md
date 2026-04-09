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


<div class="speaker-row">
  <img src="Marco.png" alt="Marco Magliulo">
  <img src="Tom.png" alt="Tom Walter">
</div>

<div class="speaker-caption">
  Marco Magliulo &nbsp;&nbsp;|&nbsp;&nbsp; Tom Walter
</div>


___

## Training Resources

- slides
- Python code
- shell launchers
 
```
https://github.com/LuxProvide/ScynergyGPUProfiling2026
```


- we will clone this repo in a few minutes on Meluxina
- you can clone it locally to get the slides 

---
#  


- Intro 
  - YOLO by ultralytics: a demo
  - Use case: fine-tuning of a YOLO model with a large dataset
- Profiling
  - Why we need profiling?
  - Where is the code running? Understand your hardware 
  - How to profile ? Intro to NVIDIA Nsight
  - Demo: profiling the sub-optimal "out-of-the-box" code
  - Hands‑on: Modifying - Profiling - Restart 
- Typical performance issues & how to fix them
- Closing words

---


---
# Agenda  


- Intro 
  - YOLO by ultralytics: a demo
  - Use case: fine-tuning of a YOLO model with a large dataset
- Profiling
  - Why we need profiling?
  - Where is the code running? Understand your hardware 
  - How to profile ? Intro to NVIDIA Nsight
  - Demo: profiling the sub-optimal "out-of-the-box" code
  - Hands‑on: Modifying - Profiling - Restart 
- Typical performance issues & how to fix them
- Closing words

---

# Goal of this workshop

By the end, you should be able to:

- Profile your GPU jobs on Meluxina
- Interpret key trace metrics and timelines  
- Identify common GPU bottlenecks (IO, compute, memory, synchronization, communication)  
- Apply simple optimizations and validate improvements  

---

<!-- _class: lead -->

# Intro 


---

# YOLO by Ultralytics 


---
# Connection to OOD 

OpenOnDemand -> https://portal.lxp.lu/

![alt text](image.png)
---


---
# Openning the Desktop app 

![alt text](image-1.png)

---
# Choosing the appropriate job options 

![alt text](image-2.png)

---
# Accessing the session

![alt text](image-3.png)

---
# Openning the terminal app

![alt text](image-4.png)


---
# Cloning the repo and setting up the environment 


```bash
$ cd $SCYNERGYDIR
$ git clone https://github.com/LuxProvide/ScynergyGPUProfiling2026
$ cd ScynergyGPUProfiling2026/
$ source setup_env.sh
```

---

# Demo: inference with object detection on a live youtube stream 

```
$ python detect_on_youtube_live_stream/test_stream_youtube_live.py
```


---
# Demo: inference with object detection on a live youtube stream 

<!-- I recorded wt Command+Shift+5 -->

<video controls src="Screen Recording 2026-04-08 at 12.40.04.mov" title="Title"></video>





<!-- ---

- Intro 
  - YOLO by ultralytics: a demo
  - Use case: fine-tuning of a YOLO model with a large dataset

# What we will be using today 

- OpenOnDemand -> https://portal.lxp.lu/
- Meluxina GPU nodes -->

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

---

# Why profile GPU code?

- Wall‑clock runtime alone doesn’t explain *why* a job is slow
- GPU programming adds complexity:
  - Host ↔ device transfers
  - Kernel launches and occupancy
  - Memory hierarchy (global / shared / L2 / registers)
- Profiling answers:
  - Is the GPU busy or idle?
  - Are we compute‑bound or memory‑bound?
  - Where is time actually spent?

---

# NVIDIA Nsight tool family

- **Nsight Systems**
  - System‑wide timeline (CPU, GPU, MPI, I/O)
  - Good for: *“Where is the time going?”*
- **Nsight Compute**
  - Kernel‑level analysis and metrics
  - Good for: *“Why is this specific kernel so slow?”*

Today focus on **Nsight Systems** 

---

# Typical profiling workflow

1. **Reproduce the problem** with a smaller test case
2. Run **Nsight Systems** to get a global view
   - CPU vs GPU overlap, kernel launches, data transfers
3. Identify **hot kernels** (top time consumers)
4. Run **Nsight Compute** on selected kernels
   - Check occupancy, memory throughput, stalls
5. Formulate hypotheses → apply changes → re‑profile
6. Repeat until performance converges

---

# Nsight Systems – what you see

- CPU thread activity and synchronization
- GPU activity:
  - Kernel launches
  - Memcpy / unified memory migrations
- Correlation with:
  - MPI, OpenMP, CUDA, libraries (cuBLAS, cuDNN, etc.)
- Key usages:
  - Detect serialization and idle GPUs
  - Spot excessive small kernel launches
  - Check overlap of communication and computation

---

# Nsight Compute – what you see

- Per‑kernel:
  - Duration, launch configuration, grid/block sizes
  - Achieved occupancy
  - Memory throughput vs. theoretical peak
  - Instruction mix (FP32/FP64, tensor cores, etc.)
  - Warp stall reasons
- Helps answer:
  - Are we limited by compute or memory?
  - Is the kernel well‑configured for the hardware?
  - Are we making good use of caches and shared memory?

---

# Running Nsight on MeluXina: concepts

- **Batch system (Slurm)**:
  - Profile inside a job allocation  
  - Nsight replaces `srun` or wraps your executable
- **Interactive vs batch**:
  - Interactive node for quick experiments
  - Batch jobs for longer profiles
- **Headless vs GUI**:
  - Run collectors on MeluXina nodes
  - Download `.qdrep` / `.nsys-rep` results and open locally in GUI

---

# Example: Nsight Systems on MeluXina

Basic pattern:

```bash
nsys profile \
  -o profile_gpu_app \
  --stats=true \
  ./your_gpu_application [args...]
```

Or inside a Slurm script:

```bash
srun nsys profile -o profile_gpu_app ./your_gpu_application
```

- Generates `profile_gpu_app.nsys-rep`
- Analyze offline with:
  - `nsys stats profile_gpu_app.nsys-rep`
  - Nsight Systems GUI on your workstation

---

# Example: Nsight Compute on MeluXina

Profile a specific kernel:

```bash
ncu --target-processes all \
    --set full \
    -o my_kernel_profile \
    ./your_gpu_application [args...]
```

- Produces `my_kernel_profile.ncu-rep`
- Inspect with:
  - `ncu-ui` locally (GUI)
  - Or `ncu --import my_kernel_profile.ncu-rep --list-sections`

> For production runs, you’ll typically narrow down to specific metrics and kernels.

---

# Common GPU performance issues

- Low GPU utilization / long CPU‑only phases
- Many tiny kernel launches (launch overhead dominated)
- Poor memory access patterns:
  - Non‑coalesced global memory
  - Excessive host↔device copies
- Low occupancy:
  - Too many registers per thread
  - Too small block sizes
- Imbalance across GPUs in multi‑GPU jobs

---

# Reading a typical Nsight timeline

Look for:

- **Gaps** on GPU lanes:
  - Is GPU idle while CPU is busy?
- **Memcpy spikes**:
  - Large or frequent data transfers?
  - Transfers overlapping with compute?
- **Kernel launch bursts**:
  - Many short kernels → consider kernel fusion or batching
- Alignment with:
  - MPI calls
  - File I/O
  - Synchronizations (barriers, `cudaDeviceSynchronize()`)

---

# From symptoms to hypotheses

Examples:

- Symptom: GPU idle, long CPU regions
  - Hypothesis: work not offloaded / blocked by synchronization
- Symptom: memcpy time dominant
  - Hypothesis: data layout / transfer strategy suboptimal
- Symptom: one kernel dominates runtime
  - Hypothesis: micro‑optimizing that kernel may yield large gains
- Symptom: low occupancy, many stalls
  - Hypothesis: tune block size, shared memory usage, loop structure

---

# Using Nsight Compute metrics

Key metrics to check (conceptually):

- **Achieved occupancy**  
- **DRAM throughput** vs peak  
- **L2 / L1 / shared memory hit rates**  
- **Warp stall reasons** (e.g. memory dependency, execution dependency, barrier)  
- **Instruction throughput** (FP32/FP64, tensor core utilization)

These guide whether you should:
- Focus on memory access patterns
- Reduce divergence / control flow issues
- Adjust launch configuration

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
