using FFTW


function initFFT(eps::Array{T,4}) where {T<:Number}
    Neps = size(eps)
    N1, N2, N3 = Neps[2:end]

    tau = zeros(Complex{T}, 6, div(N1, 2) + 1, N2, N3)

    #? FFT plans
    P = plan_rfft(eps, (2, 3, 4))
    Pinv = plan_irfft(tau, N1, (2, 3, 4))

    if T == Float64 || T == ComplexF64
        C = Float64
    else
        C = Float32
    end

    xi1 = zeros(C, div(N1, 2) + 1)
    xi2 = zeros(C, N2)
    xi3 = zeros(C, N3)
    xi1 .= rfftfreq(N1, N1)
    xi2 .= fftfreq(N2, N2)
    xi3 .= fftfreq(N3, N3)

    return P, Pinv, xi1, xi2, xi3, tau

    return P, Pinv, xi1, xi2, xi3, tau
end


function initFFT(eps::CuArray{T,4}) where {T<:Number}
    Neps = size(eps)
    N1, N2, N3 = Neps[2:end]

    tau = CUDA.zeros(Complex{T}, 6, div(N1, 2) + 1, N2, N3)

    #? FFT plans
    P = CUDA.CUFFT.plan_rfft(eps, (2, 3, 4))
    Pinv = CUDA.CUFFT.plan_irfft(tau, N1, (2, 3, 4))

    if T == Float64 || T == ComplexF64
        C = Float64
    else
        C = Float32
    end

    xi1 = CUDA.zeros(C, div(N1, 2) + 1)
    xi2 = CUDA.zeros(C, N2)
    xi3 = CUDA.zeros(C, N3)
    xi1 .= CUDA.CUFFT.rfftfreq(N1, N1)
    xi2 .= CUDA.CUFFT.fftfreq(N2, N2)
    xi3 .= CUDA.CUFFT.fftfreq(N3, N3)

    return P, Pinv, xi1, xi2, xi3, tau
end


function get_blocks_threads(x)
    threads_per_block = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    n_blocks = cld(length(x), threads_per_block)
    return n_blocks, threads_per_block
end

function get_blocks_threads(N::Int64)
    threads_per_block = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    n_blocks = cld(N, threads_per_block)
    return n_blocks, threads_per_block
end


function add_mean_value_kernel!(eps, mean_value, NNN, cartesian)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= NNN
        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        eps[1, i1, i2, i3] += mean_value[1]
        eps[2, i1, i2, i3] += mean_value[2]
        eps[3, i1, i2, i3] += mean_value[3]
        eps[4, i1, i2, i3] += mean_value[4]
        eps[5, i1, i2, i3] += mean_value[5]
        eps[6, i1, i2, i3] += mean_value[6]
    end
    return nothing
end


function add_mean_value!(eps::CuArray, mean_value::Vector, cartesian)
    mean_value_gpu = cu(mean_value)
    NNN = size(eps, 1) * size(eps, 2) * size(eps, 3)
    n_blocks, n_threads = get_blocks_threads(NNN)
    @cuda blocks = n_blocks threads = n_threads add_mean_value_kernel!(eps, mean_value_gpu, NNN, cartesian)
end

function add_mean_value!(eps::Array, mean_value::Vector, args...)
    eps[1, :, :, :] .+= mean_value[1]
    eps[2, :, :, :] .+= mean_value[2]
    eps[3, :, :, :] .+= mean_value[3]
    eps[4, :, :, :] .+= mean_value[4]
    eps[5, :, :, :] .+= mean_value[5]
    eps[6, :, :, :] .+= mean_value[6]
end