![SCynergy 2026](./assets/scynergy2026.png){ width="640" }

# Understanding why your GPU-accelerated is slow using NVIDIA Nsight Systems

This training is given in the context of the [Scynergy 2026 event](https://www.scynergy.events/). It provides participants with a hands-on introduction to profiling using NVIDIA NSigh-Systems on the [MeluXina supercomputer](https://docs.lxp.lu/system/overview/).

![](images/meluxina.jpg){ width="800" }

## 🎯 Objectives

By the end of this workshop, you should be able to:

- Profile your GPU jobs on MeluXina
- Interpret key NSight-Systems trace metrics and timelines  
- Identify common GPU bottlenecks (IO, compute, memory, synchronization, communication)  
- Apply simple optimizations and validate improvements  

## 🪧 Agenda

Today's training is composed of:

- Connection to MeluXina via OpenOnDemand (10 minutes)
- Introduction to NVIDIA NSight-Systems (~30 minutes)
    - Looking at (already collected) traces 
    - Exploring `nsys-ui` and the `nsys` CLI
- Hands-on: making a PyTorch Training faster (~60 minutes)
    - Profile and Analysze
    - Modify the code to accelerate it

---

## 💻 Hands-on: Getting Started with MeluXina

- [Hands-on Part 1: 🔑 Settings things up](setup.md)
- [Hands-on Part 2: 😩️ Profile a slow code](nsight_systems.md) 
- [Hands-on Part 3: 💡 Optimizing your code](optimizing.md)


## ℹ️ About this training

It has been developed by the **Supercomputing Application Services** group at [**LuxProvide**](https://luxprovide.lu) in the context of the [**EPICURE project**](https://epicure-hpc.eu/).


<div class="speaker-row">
  <img src="Marco.png" alt="Marco Magliulo">
  <img src="Tom.png" alt="Tom Walter">
</div>

<div class="speaker-caption">
  Marco Magliulo &nbsp;&nbsp;|&nbsp;&nbsp; Tom Walter
</div>


[![EPICURE](./assets/logo_epicure.png){ width="420" }](https://epicure-hpc.eu/) 
[![LuxProvide](./assets/logo_luxprovide.png){ width="320" }](https://luxprovide.lu)
