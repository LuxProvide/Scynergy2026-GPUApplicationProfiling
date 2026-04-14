
# Your turn to look at a trace 

This time, you are in command.
You will do two things:
- open another trace and comment on it
- profile a better performing code, open the trace, experiment what happens when the parameters are being changed 


## What if we try something brute force and just increase the number of GPUs used i? 

---


run the following: 

```bash
nsys-ui $TRACE_4GPU_BASE
```

Let's take 10 minutes for you to play around with this trace and then we will debrief  

![alt text](images/image-12.png)

---
## Observations

- We knew that the problem was not coming from the GPU usage when using 1 GPU 
- Still, we wanted to see if using 4 GPUs would reduce the training time 
- still a lot of gaps in the individual activity of the GPU in the distributed training
- single GPU: 217 sec for the epoch
- 4 GPUs: 179 sec

⚠️ 4x more GPU power but 18% improvement in runtime 
⚠️ My ennemy is still the same: the dataloader  

---

<!-- _class: lead -->
## What can kill the performance that much ? 

<!-- _class: lead -->

### Your turn to investigate 

- This time you are totally in command. You will
    * collect a trace 
    * analyze it
    * find what is/are the bottleneck(s)
    * modify the code accordingly

---

### Starting point 

```bash
cd /project/home/p201259/workspaces/$USER/Scynergy2026-GPUApplicationProfiling/Script
```

---
### Code to use 

- Python script: `script_modded_4g.py`
- Launcher: `source launcher_modded_4g_p.sh`

To launch the script **from the OpenOnDemand** terminal:

```bash
source launcher_modded_4g_p.sh 
```

---
### Openning the trace

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

## Conclusion

### Example of trace: improved execution

![alt text](images/image-14.png)

---

### Wrapping things up 

- 1 GPU - base script - 217 seconds
- 4 GPUs - base script - 189 seconds
- 4 GPUs - improved script - 9 seconds 

---

### Recap of the workflow when you need to improve your code performance 

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
    nsys-profile ${NSYS_OPTIONS} ${YOURBIN}
    ```

    or if `srun` is available (outside of OpenOnDemand): 

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

<!-- 
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

![width:1000px](/docs/images/image-15.png) -->