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

---
---

## Agenda 


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

<!-- _class: lead -->

# Why profiling? 

---

# Why Measuring/profiling GPU code?

- Wall‑clock runtime alone doesn’t explain *why* a job is slow
- GPU programming adds complexity:
  - Host ↔ device transfers
  - Kernel launches and occupancy
  - Memory hierarchy (global / shared / L2 / registers)


---
##  Typical Key Questions Answered via Profiling

- **CPU/IO Bottlenecks:** Is the GPU idle during data loading?
- **Compute vs. Memory:** bandwidth-bound or compute-bound?
- **Multi-GPU Scaling:** Is there load imbalance across ranks?
- **Sync Stalls:** Are MPI/NCCL/Barriers causing idleness?
- **Launch Overhead:** Are kernels too small or frequent?
- **Occupancy:** Is register/SRAM usage limiting parallelism?

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

# What we will be using  

For the ease of use, we are going to use OpenOnDemand.
We will then be able to:
- Open `Nsight-Systems` GUI to analyze traces already prepared for you,
- We will also use the `nsys` command line tools

---

<!-- _class: lead -->

# Let's start  

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

- Here we look at the trace corresponding to the code `Scripts/script_basic_1g.py`
- This is what you would get from a "naive" training code

--- 
# Second step: let's open a trace

![width:900px](<Screenshot 2026-04-11 at 13.04.13.png>)

___

# Let's have a closer look

![width:800px](image-5.png)


---

# Zoom on a part of the timeline

Hover your mouse over a refion of interest by keeping the left button of your mouse pressed.


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


![width:500px](image-9.png)

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

In the GUI, you can select the analysis summary allows you to retrieve which command line you used to obtain the trace.
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

# Alternatives 

For collection, you can also reduce trace size and overhead with `--delay` and/or `--duration`


--- 

# Nsight Systems CLI

- once you have your `.nsys-rep`, you can also use the CLI to post-process the profiling output
- `nsys` can post-process existing `.nsys-rep` or SQLite results using `stats`, `analyze`, `export`, and `recipe`

---

# Nsight Systems CLI

- `nsys stats` → generate statistical summaries 
- `nsys analyze` → generate an expert-systems report 
- `nsys export` → generate an export file from an existing `.nsys-rep`. 
- `nsys recipe` → post-process **multiple existing results** to generate statistics and various plots. 

---

# Fastest starting point: `nsys stats`

```bash
nsys stats report.nsys-rep
```

*   `nsys stats` is the quickest way to get useful text summaries from a saved report. 
*   It accepts either `.nsys-rep` or SQLite input.

---

# Ask targeted questions with report scripts

```bash
nsys stats --report cuda_api_sum report.nsys-rep
nsys stats --report cuda_gpu_kern_sum report.nsys-rep
nsys stats --report cuda_gpu_trace report.nsys-rep
```

---

*   Standard report scripts shipped with Nsight Systems. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   `cuda_api_sum` summarizes CUDA API calls and their execution times. 
*   `cuda_gpu_kern_sum` summarizes CUDA kernels and their execution times. 
*   `cuda_gpu_trace` prints a trace of CUDA kernels and memory operations sorted by start time. 

***

# Useful report variants

```bash
nsys stats --report cuda_gpu_kern_sum:nvtx-name report.nsys-rep
nsys stats --report cuda_gpu_kern_gb_sum:base report.nsys-rep
nsys stats --report cuda_api_gpu_sum:mangled report.nsys-rep
```

*   Several standard reports support optional modifiers such as `:nvtx-name`, `:base`, and `:mangled`. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   `:nvtx-name` prefixes a kernel with the name of the innermost enclosing NVTX range. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   `:base` aggregates by base kernel name rather than the fully templated name, while `:mangled` uses the raw mangled name when available. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

***

# Multiple reports and machine-readable output

```bash
nsys stats \
  --report cuda_gpu_trace \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  --format csv,column \
  --output .,- \
  report.nsys-rep
```

*   `nsys stats` can generate multiple reports in one invocation. [\[docs.nvidia.com\]](https://docs.nvidia.com/drive/drive-os-5.2.3.0L/nsight-systems/pdf/UserGuide.pdf), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
*   It can also emit different output formats, which is useful for terminal inspection and scripting. [\[docs.nvidia.com\]](https://docs.nvidia.com/drive/drive-os-5.2.3.0L/nsight-systems/pdf/UserGuide.pdf), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
*   A practical pattern is to keep one human-readable view and one CSV export for automation. [\[docs.nvidia.com\]](https://docs.nvidia.com/drive/drive-os-5.2.3.0L/nsight-systems/pdf/UserGuide.pdf), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

***

# `nsys analyze` and `nsys export`

```bash
nsys analyze report.nsys-rep
nsys export report.nsys-rep
```

*   `nsys analyze` is the expert-systems layer: it post-processes an existing result and generates an expert report. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/2024.3/UserGuide/index.html)
*   `nsys export` is the conversion layer: it generates export files from an existing `.nsys-rep`. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/2024.3/UserGuide/index.html)
*   A simple mental model is: `stats` for summaries, `analyze` for guidance, and `export` for downstream tooling. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

***

# What `nsys recipe` is for

*   `nsys recipe` is intended for **post-processing multiple existing results** rather than just one report. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/2024.3/UserGuide/index.html)
*   Its output is higher-level statistical information and various plots. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   The Analysis Guide explicitly notes that recipes can use `--timeunit` to change output time units from the default nanoseconds. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

***

# `nsys recipe` — minimal template

```bash
nsys recipe <recipe-name> <recipe-options> --timeunit ms
```

*   The key idea is that a recipe runs a named post-processing workflow on existing profiling results. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   Recipes are meant for multi-report analysis and plot generation, so they are a good fit for benchmark campaigns and regression studies. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   `--timeunit` is explicitly documented as a recipe option for changing time units. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

***

# `nsys recipe` — community example

```bash
nsys recipe nccl_gpu_overlap_trace \
  --input ./profile/ \
  --output ./output/nccl_gpu_overlap_trace
```

*   A community-maintained `nsys_recipes` repository shows a concrete `nsys recipe` invocation pattern using a recipe name plus `--input` and `--output`. [\[github.com\]](https://github.com/hyxcl/nsys_recipes/blob/main/README.md), [\[github.com\]](https://github.com/hyxcl/nsys_recipes)
*   That repository describes its recipes as a supplement to Nsight Systems’ built-in multi-report recipe support. [\[github.com\]](https://github.com/hyxcl/nsys_recipes/blob/main/README.md), [\[github.com\]](https://github.com/hyxcl/nsys_recipes)
*   One documented example is `nccl_gpu_overlap_trace`, which focuses on communication/compute overlap analysis. [\[github.com\]](https://github.com/hyxcl/nsys_recipes/blob/main/README.md), [\[github.com\]](https://github.com/hyxcl/nsys_recipes)

***

# Why recipes matter

*   `stats` is excellent for **single-report summaries**. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   `recipe` becomes more interesting when you have **many reports** and want **comparative statistics or plots**. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
*   This makes `recipe` a natural fit for parameter sweeps, nightly performance baselines, and multi-run studies. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

***

# Practical workflow

1.  Start with `nsys stats report.nsys-rep` for quick triage. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
2.  Use targeted reports such as `cuda_api_sum`, `cuda_gpu_kern_sum`, or `cuda_kern_exec_sum` to answer specific questions. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
3.  Move to `nsys analyze` if you want expert-style guidance. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/2024.3/UserGuide/index.html)
4.  Use `nsys export` or `nsys recipe` when you need downstream tooling, comparative analysis, or plots across multiple reports. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

***

# Minimal cheat sheet

```bash
# quick summary
nsys stats report.nsys-rep

# focused summaries
nsys stats --report cuda_api_sum report.nsys-rep
nsys stats --report cuda_gpu_kern_sum report.nsys-rep
nsys stats --report cuda_kern_exec_sum report.nsys-rep

# expert report
nsys analyze report.nsys-rep

# exports
nsys export report.nsys-rep

# multi-report analysis
nsys recipe <recipe-name> <recipe-options> --timeunit ms
```

*   The takeaway is simple: **profile once, post-process many times**. [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [\[docs.nvidia.com\]](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

```




---


<!-- _class: lead -->

# Your turn to look at a trace 

---

<!-- _class: lead -->

What if we try something brute force and just increase the number of GPUs used i? 

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

- We knew that the problem was not coming from the GPU usage when using 1 GPU 
- Still, we wanted to see if using 4 GPUs would reduce the training time 
- still a lot of gaps in the individual activity of the GPU in the distributed training
- single GPU: 217 sec for the epoch
- 4 GPUs: 179 sec

⚠️ 4x more GPU power but 18% improvement in runtime 

---

<!-- _class: lead -->
## What can kill the performance that much ? 

<!-- _class: lead -->

# Your turn to investigate 

---

## Starting point 

```bash
cd /project/home/p201259/workspaces/$USER/Scynergy2026-GPUApplicationProfiling/Script
```

---
## Code to use 

- Python script: `script_modded_4g.py`
- Launcher: `source launcher_modded_4g_p.sh`

To launch the script **from the OpenOnDemand** terminal:

```bash
source launcher_modded_4g_p.sh 
```

---
## Openning the trace

```bash
sys-rep
train completed, best_metric: 0.8383 at epoch: 1
Generated:
        .../modded_4g_no_p.mel2129.24037.nsys-rep
```

Get the path of the ``.nsys-rep`` file and open it with:

```bash
nsys-ui $THEPATH.nsys-rep
```

---

## Example of improved script trace 

![width:1200px](image-14.png)

---

## Wrapping things up 

- 1 GPU - base script - 217 seconds
- 4 GPUs - base script - 189 seconds
- 4 GPUs - improved script - 9 seconds 

---

# Recap of the workflow when you need to improve your code performance 

1. Prepare a **representative but smaller** test case if the code is too long to execute  
2. Run Nsight Systems for a **global view** on the base script
3. Identify the 2–3 main bottlenecks from the trace
4. Implement optimizations → re‑run profiling 
5. Carefully review what you have changed. Try not to change only one thing at a time 
6. Ensure that your modifications did not affect the code functionnality (for example convergence of training)
7. Once satisfied, scale back up to full production sizes. 


---

# What you can do after this workshop

- Apply Nsight to your **own applications** on MeluXina to produce traces  

```bash
srun ${SRUN_OPTIONS} nsys-profile ${NSYS_OPTIONS} ${YOURBIN}
```

- Performance optimization guided by profiling results 

---

# Q & A

Questions, specific applications, or issues you’d like to discuss?

---

# Thank you

- Useful resources:
  - [NSight documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
  - [Meluxina Documentation](https://docs.lxp.lu/)

- Contact:  
  - servicedesk [at] lxp.lu


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

---

# Zoom on a dataloading part of the training 

![width:1000px](image-15.png)