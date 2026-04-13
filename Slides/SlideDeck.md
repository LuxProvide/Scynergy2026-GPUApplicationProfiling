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

![width:800px](/docs/images/image-12.png)

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

![width:1200px](/docs/images/image-14.png)

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

![width:1000px](/docs/images/image-15.png)