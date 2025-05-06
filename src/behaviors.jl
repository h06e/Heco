using Parameters
using CUDA
using Adapt

abstract type Elastic end

export IE
export ITE
export IE2ITE
export convert_mat


#**************************************************************

struct IE{T<:Number} <: Elastic
    kappa::T
    mu::T
    E::T
    nu::T
    lambda::T
end
Adapt.@adapt_structure IE


function IE(kappa::T, mu::T) where {T<:Number}
    E = 9.0 * kappa * mu / (3.0 * kappa + mu)
    nu = E / 2.0 / mu - 1.0
    lambda = kappa - 2.0 / 3.0 * mu
    return IE{T}(kappa, mu, E, nu, lambda)
end

function IE(; kwargs...)
    if haskey(kwargs, :E) && haskey(kwargs, :nu)
        kappa = kwargs[:E] / 3.0 / (1 - 2 * kwargs[:nu])
        mu = kwargs[:E] / 2 / (1 + kwargs[:nu])
        return IE(kappa, mu)
    elseif haskey(kwargs, :E) && haskey(kwargs, :mu)
        kappa = kwargs[:E] * kwargs[:mu] / 3 / (3 * kwargs[:mu] - kwargs[:E])
        return IE(kappa, kwargs[:mu])
    elseif haskey(kwargs, :kappa) && haskey(kwargs, :mu)
        return IE(kwargs[:kappa], kwargs[:mu])
    elseif haskey(kwargs, :lambda) && haskey(kwargs, :mu)
        kappa = kwargs[:lambda] + 2.0 / 3.0 * kwargs[:mu]
        return IE(kappa, kwargs[:mu])
    else
        @error "Bad arguments for IE material definition"
        throw(ArgumentError)
    end
end

function eigvals_mat(mat::IE)
    return 3 * mat.kappa, 2 * mat.mu
end

function IE2ITE(mat::IE)
    ITE(; k=mat.lambda + mat.mu,
        l=mat.lambda,
        m=mat.mu,
        n=mat.lambda + 2 * mat.mu,
        p=mat.mu)
end

function Base.:+(m1::Elastic, m2::Elastic)
    m3 = IE2ITE(m1)
    m4 = IE2ITE(m2)
    ITE(k=m4.k + m3.k, l=m4.l + m3.l, m=m4.m + m3.m, n=m4.n + m3.n, p=m4.p + m3.p)
end


#**************************************************************

struct ITE{T<:Number} <: Elastic
    k::T
    l::T
    m::T
    n::T
    p::T
    El::T
    Et::T
    nul::T
    nut::T
    mul::T
    mut::T
end
Adapt.@adapt_structure ITE

function ITE(k::T, l::T, m::T, n::T, p::T) where {T<:Number}
    El = n - l * l / k
    Et = 4 * m * k * El / (m * n + k * El)
    nul = l / 2.0 / k
    nut = (k * El - m * n) / (k * El + m * n)
    mut = Et / 2.0 / (1 + nut)
    mul = p

    ITE{T}(k, l, m, n, p, El, Et, nul, nut, mul, mut)
end

function ITE(; kwargs...)
    if all(key -> haskey(kwargs, key), [:k, :l, :m, :n, :p])
        @unpack k, l, m, n, p = kwargs
        return ITE(k, l, m, n, p)

    elseif all(key -> haskey(kwargs, key), [:El, :Et, :nul, :nut, :mul])
        @unpack El, Et, nul, nut, mul = kwargs
        p = mul
        deno = (El - El * nut - 2 * Et * nul^2)
        n = El^2 * (1 - nut) / deno
        l = El * Et * nul / deno
        kpm = Et * (El - Et * nul^2) / (1 + nut) / deno
        kmm = Et * (El * nut + Et * nul^2) / (1 + nut) / deno
        k = 0.5 * (kpm + kmm)
        m = 0.5 * (kpm - kmm)
        return ITE(; k=k, l=l, m=m, n=n, p=p)

    elseif all(key -> haskey(kwargs, key), [:El, :Et, :nul, :mut, :mul])
        @unpack El, Et, nul, mut, mul = kwargs
        nut = Et / 2 / mut - 1
        return ITE(; El=El, Et=Et, nul=nul, nut=nut, mul=mul)
    else
        @error "Bad arguments for ITE material definition"
        throw(ArgumentError)
    end
end

function eigvals_mat(mat::ITE)
    v1 = 2 * mat.m
    v2 = 2 * mat.p
    i = mat.k + mat.n / 2.0
    j =
        0.5 *
        sqrt(4 * mat.k * mat.k - 4 * mat.k * mat.n + 8 * mat.l * mat.l + mat.n * mat.n)
    v3 = i + j
    v4 = i - j
    return v1, v2, v3, v4
end


function IE2ITE(mat::ITE)
    mat
end


function convert_mat(mat::ITE, T::DataType)
    return ITE(k=T(mat.k), l=T(mat.l), m=T(mat.m), n=T(mat.n), p=T(mat.p))
end

function convert_mat(mat::IE, T::DataType)
    return IE(kappa=T(mat.kappa), mu=T(mat.mu))
end


function convert_precision(mat::IE{T}, precison::Precision) where {T<:Number}
    if precison == Simple
        if T <: Real
            return convert_mat(mat, Float32)
        else
            return convert_mat(mat, ComplexF32)
        end
    else
        if T <: Real
            return convert_mat(mat, Float64)
        else
            return convert_mat(mat, ComplexF64)
        end
    end
end

function convert_precision(mat::ITE{T}, precison::Precision) where {T<:Number}
    if precison == Simple
        if T <: Real
            return convert_mat(mat, Float32)
        else
            return convert_mat(mat, ComplexF32)
        end
    else
        if T <: Real
            return convert_mat(mat, Float64)
        else
            return convert_mat(mat, ComplexF64)
        end
    end
end

function convert_precision(vect_mat::Vector{<:Elastic}, precision::Precision)
    return [convert_precision(m, precision) for m in vect_mat]
end

#************************************************************************************
#* CPU
#************************************************************************************



function rdc!(
    sig::Array{T,4},
    eps::Array{T,4},
    phases::Array{<:Integer,3},
    material_list::Vector{<:Elastic},
    args...
) where {T<:Number}
    N1, N2, N3, _ = size(eps)

    @inbounds for k in 1:N3, j in 1:N2, i in 1:N1
        mat = material_list[Int(phases[i, j, k])]

        k_, m, l, n, p = mat.k, mat.m, mat.l, mat.n, mat.p

        e1, e2, e3 = eps[i, j, k, 1], eps[i, j, k, 2], eps[i, j, k, 3]
        sig[i, j, k, 1] = (k_ + m) * e1 + (k_ - m) * e2 + l * e3
        sig[i, j, k, 2] = (k_ - m) * e1 + (k_ + m) * e2 + l * e3
        sig[i, j, k, 3] = l * e1 + l * e2 + n * e3
        sig[i, j, k, 4] = 2 * p * eps[i, j, k, 4]
        sig[i, j, k, 5] = 2 * p * eps[i, j, k, 5]
        sig[i, j, k, 6] = 2 * m * eps[i, j, k, 6]
    end
end

function rdc_inv!(eps::Array{T,4}, sig::Array{T,4}, phases::Array{<:Integer,3}, material_list::Vector{<:Elastic}) where {T<:Number}

    N1, N2, N3, _ = size(eps)
    @inbounds for k in 1:N3, j in 1:N2, i in 1:N1

        mat = material_list[phases[i, j, k]]

        s11 = 1 / mat.Et
        s33 = 1 / mat.El
        s12 = -mat.nut / mat.Et
        s13 = -mat.nul / mat.El
        s44 = 1.0 / 2.0 / mat.mul
        s66 = 1.0 / 2.0 / mat.mut

        eps[i, j, k, 1] = s11 * sig[i, j, k, 1] + s12 * sig[i, j, k, 2] + s13 * sig[i, j, k, 3]
        eps[i, j, k, 2] = s12 * sig[i, j, k, 1] + s11 * sig[i, j, k, 2] + s13 * sig[i, j, k, 3]
        eps[i, j, k, 3] = s13 * sig[i, j, k, 1] + s13 * sig[i, j, k, 2] + s33 * sig[i, j, k, 3]
        eps[i, j, k, 4] = s44 * sig[i, j, k, 4]
        eps[i, j, k, 5] = s44 * sig[i, j, k, 5]
        eps[i, j, k, 6] = s66 * sig[i, j, k, 6]


    end
end



function compute_sig(eps::Vector{<:Number}, mat::IE)
    tre = eps[1] + eps[2] + eps[3]
    sig = [
        2 * mat.mu * eps[1] + mat.lambda * tre,
        2 * mat.mu * eps[2] + mat.lambda * tre,
        2 * mat.mu * eps[3] + mat.lambda * tre,
        2 * mat.mu * eps[4],
        2 * mat.mu * eps[5],
        2 * mat.mu * eps[6],
    ]
    return sig
end

function compute_sig(eps::Vector{<:Number}, mat::ITE)
    sig = [
        (mat.k + mat.m) * eps[1] + (mat.k - mat.m) * eps[2] + (mat.l) * eps[3],
        (mat.k - mat.m) * eps[1] + (mat.k + mat.m) * eps[2] + (mat.l) * eps[3],
        (mat.l) * eps[1] + (mat.l) * eps[2] + (mat.n) * eps[3],
        2 * mat.p * eps[4],
        2 * mat.p * eps[5],
        2 * mat.m * eps[6],
    ]
    return sig
end



function compute_eps(sig::Vector{<:Number}, mat::IE)
    trs = sig[1] + sig[2] + sig[3]
    unpnusE = (1 + mat.nu) / (mat.E)
    mnusE = -mat.nu / mat.E
    eps = [
        unpnusE * sig[1] + mnusE * trs,
        unpnusE * sig[2] + mnusE * trs,
        unpnusE * sig[3] + mnusE * trs,
        unpnusE * sig[4],
        unpnusE * sig[5],
        unpnusE * sig[6],
    ]
    return eps
end

function compute_eps(sig::Vector{<:Number}, mat::ITE)
    s11 = 1 / mat.Et
    s33 = 1 / mat.El
    s12 = -mat.nut / mat.Et
    s13 = -mat.nul / mat.El
    s44 = 1.0 / 2.0 / mat.mul
    s66 = 1.0 / 2.0 / mat.mut

    eps = [
        s11 * sig[1] + s12 * sig[2] + s13 * sig[3],
        s12 * sig[1] + s11 * sig[2] + s13 * sig[3],
        s13 * sig[1] + s13 * sig[2] + s33 * sig[3],
        s44 * sig[4],
        s44 * sig[5],
        s66 * sig[6],
    ]
    return eps
end



#**********************************************************************************
#* GPU
#**********************************************************************************

function rdc!(sig::CuArray, eps::CuArray, phases::CuArray, material_list, cartesian)
    NNN = length(phases)
    n_blocks, n_threads = get_blocks_threads(phases)
    @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, eps, phases, material_list, cartesian, NNN)
end


function rdcgpu!(sig, eps, phases, material_list, cartesian, NNN)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= NNN

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        mat = material_list[phases[i]]

        sig[i1, i2, i3, 1] = (mat.k + mat.m) * eps[i1, i2, i3, 1] + (mat.k - mat.m) * eps[i1, i2, i3, 2] + mat.l * eps[i1, i2, i3, 3]
        sig[i1, i2, i3, 2] = (mat.k - mat.m) * eps[i1, i2, i3, 1] + (mat.k + mat.m) * eps[i1, i2, i3, 2] + mat.l * eps[i1, i2, i3, 3]
        sig[i1, i2, i3, 3] = mat.l * eps[i1, i2, i3, 1] + mat.l * eps[i1, i2, i3, 2] + mat.n * eps[i1, i2, i3, 3]
        sig[i1, i2, i3, 4] = 2 * mat.p * eps[i1, i2, i3, 4]
        sig[i1, i2, i3, 5] = 2 * mat.p * eps[i1, i2, i3, 5]
        sig[i1, i2, i3, 6] = 2 * mat.m * eps[i1, i2, i3, 6]

    end
    return nothing
end


function rdcinvgpu!(eps, sig, phases, material_list, cartesian, NNN)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= NNN
        mat = material_list[phases[i]]

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        s11 = 1 / mat.Et
        s33 = 1 / mat.El
        s12 = -mat.nut / mat.Et
        s13 = -mat.nul / mat.El
        s44 = 1.0 / 2.0 / mat.mul
        s66 = 1.0 / 2.0 / mat.mut

        eps[i1, i2, i3, 1] = s11 * sig[i1, i2, i3, 1] + s12 * sig[i1, i2, i3, 2] + s13 * sig[i1, i2, i3, 3]
        eps[i1, i2, i3, 2] = s12 * sig[i1, i2, i3, 1] + s11 * sig[i1, i2, i3, 2] + s13 * sig[i1, i2, i3, 3]
        eps[i1, i2, i3, 3] = s13 * sig[i1, i2, i3, 1] + s13 * sig[i1, i2, i3, 2] + s33 * sig[i1, i2, i3, 3]
        eps[i1, i2, i3, 4] = s44 * sig[i1, i2, i3, 4]
        eps[i1, i2, i3, 5] = s44 * sig[i1, i2, i3, 5]
        eps[i1, i2, i3, 6] = s66 * sig[i1, i2, i3, 6]

    end
    return nothing
end