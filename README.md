
# Heco.jl

**Heco.jl** (Homogenization of Elastic COmposite materials) is a Julia package for performing homogenization computations on multi-phase materials. It includes both CPU and GPU-accelerated implementations.

While the package is generally applicable to various materials, its development was driven by the need for features specific to unidirectional composites with transversely isotropic fibers (e.g., carbon fibers).

If you find **Heco.jl** useful in your research or projects, please consider citing the following references (or see `CITATION.bib`)

---

## Installation

First, install [Julia](https://julialang.org/).

To install **Heco.jl**, run the following command in the Julia REPL:

```julia
julia> ]add "https://github.com/h06e/Heco.jl"
```

> **Note:** GPU acceleration requires an NVIDIA GPU with CUDA support. If no compatible GPU is available, the package can still be used via the CPU implementation. Some warning messages may appear when running in CPU-only mode.



📘 **Recommendation:**  
For a complete and interactive walkthrough, download the notebook available at:  
`examples/basics.ipynb`  

It presents a full example of using `Heco.jl`, including microstructure generation, material setup, homogenization, and result visualization.


---


## Features

### Microstructure Generation

**Heco.jl** includes a basic Representative Volume Element (RVE) generator tailored for continuous fiber composites. The generator uses a particle-based method, which is particularly efficient for high fiber volume fractions.

![RVE Generation](ressources/rve_gen.gif)

### Material Behavior

The package currently supports linear elastic materials:
- Isotropic
- Transversely isotropic (with the symmetry axis aligned with the 3rd axis)

Elasticity tensors can be either real or complex-valued. Complex tensors are typically used for modeling viscoelastic behavior in the Laplace-Carson domain.

### FFT-Based Solver

The solver is based on the classical Moulinec and Suquet (1994) scheme. Key features include:
- Strain-controlled or stress-controlled loading (non-mixed)
- Isotropic or transversely isotropic reference media (aligned along axis 3)
- Convergence criteria following Bellis and Suquet (2019), applied to both loading and residuals


