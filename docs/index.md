# Understanding why your GPU-accelerated is slow using NVIDIA Nsight Systems

![SCynergy 2026](./assets/scynergy2026.png)

This training is given in the context of the [Scynergy 2026 event](https://www.scynergy.events/). It provides participants with a hands-on introduction to profiling using NVIDIA NSigh-Systems on the [MeluXina supercomputer](https://docs.lxp.lu/system/overview/).

![MeluXina](images/meluxina.jpg){ width="800" }

## 🎯 Objectives

By the end of this workshop, you should be able to:

- Profile your GPU jobs on MeluXina
- Interpret key NSight-Systems trace metrics and timelines  
- Identify common bottlenecks in GPU accelerated codes/applications (IO, compute, memory, synchronization, communication)  
- Apply simple optimizations and validate improvements  

## 🪧 Agenda

Today's training is composed of:

- Connection to MeluXina via OpenOnDemand (~10 minutes)
- Introduction to NVIDIA NSight-Systems (~30 minutes)
  - How to generate a trace with `nsys-profile`
  - Looking at (already collected) traces of a [MonAI](https://project-monai.github.io/) training on Meluxina  
    - How to navigate on the NSight-Systems GUI
    - How to use the `nsys` CLI to get some stats
- Hands-on: making a PyTorch Training faster (~60 minutes)
  - Generate your own traces
  - Modify the code to accelerate it

---

## 💻 Demo/Hands-on Mix

- [Hands-on Part: Settings things up](setup.md)
- [Demo and Discussion: ️Getting to know the tool and profiling of a slow [MonAI](https://project-monai.github.io/) training code](nsight_systems.md)
- [Hands-on Part: Profile and optimize a distributed GPU-accelerated code](optimizing.md)

## ℹ️ About this training

**Authors:** **Marco Magliulo**, **Emmanuel Kieffer**, and **Tom Walter**

This training has been developed by the **Supercomputing Application Services** group at [**LuxProvide**](https://luxprovide.lu) in the context of the [**EPICURE project**](https://epicure-hpc.eu/).

[![EPICURE](./assets/logo_epicure.png){ width="420" }](https://epicure-hpc.eu/)
[![LuxProvide](./assets/logo_luxprovide.png){ width="320" }](https://luxprovide.lu)
