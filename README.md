

# Heco.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15349930.svg)](https://doi.org/10.5281/zenodo.15349930)

**Heco.jl** (Homogenization of Elastic COmposite materials) is a Julia package for performing FFT-based linear homogenization of multi-phase elastic materials. It supports both real and complex materials, with efficient implementations for CPU and GPU architectures.

Although **Heco.jl** is applicable to a wide range of materials, it was initially developed with a focus on unidirectional composites featuring transversely isotropic fibers—such as carbon-fiber-reinforced systems.

If you use **Heco.jl** in your work, please consider citing the following:

> 📖 Valmalette, R. (2025). *Heco.jl: A Real and Complex CPU–GPU Solver for FFT-Based Linear Homogenization*. Zenodo. [https://doi.org/10.5281/zenodo.15349930](https://doi.org/10.5281/zenodo.15349930)

---

## ⚙️ Installation

First, install [Julia](https://julialang.org/).

To install **Heco.jl**, run the following command in the Julia REPL:

```julia
julia> ]add "https://github.com/h06e/Heco"
```

> [!tip]
> GPU acceleration requires an NVIDIA GPU with CUDA support. If no compatible GPU is available, the package can still be used via the CPU implementation. Some warning messages may appear when running in CPU-only mode.


📘 **Recommendation:**  
For a complete and interactive walkthrough, download the notebook available at:  
`examples/basics.ipynb`  

It presents a full example of using `Heco.jl`, including microstructure generation, material setup, homogenization, and result visualization.


---


## ✨ Features


### 🧩 Microstructure Generation

**Heco.jl** includes a basic Representative Volume Element (RVE) generator tailored for continuous fiber composites. The generator uses a particle-based method, which is particularly efficient for high fiber volume fractions.

![RVE Generation](ressources/rve_gen.gif)

### 🛠️ Material Behavior

The package currently supports linear elastic materials:
- Isotropic
- Transversely isotropic (with the symmetry axis aligned with the 3rd axis)

Elasticity tensors can be either real or complex-valued. Complex tensors are typically used for modeling viscoelastic behavior in the Laplace-Carson domain.

### ⚡FFT-Based Solver

The solver is built upon the foundational Moulinec and Suquet (1994) scheme. Its key features include:
- Support for strain-controlled or stress-controlled loading (non-mixed modes).
- Compatibility with isotropic or transversely isotropic reference media (aligned along the 3rd axis).
- Convergence criteria based on the methods of Bellis and Suquet (2019), as well as stress-criteria criteria for stress-controlled loading.


