

function meanfield(x::Array{<:Number,4})
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    x5 = 0.0
    x6 = 0.0

    N1, N2, N3, _ = size(x)
    @inbounds for i3 in 1:N3, i2 in 1:N2, i1 in 1:N1
        x1 += x[i1, i2, i3, 1]
        x2 += x[i1, i2, i3, 2]
        x3 += x[i1, i2, i3, 3]
        x4 += x[i1, i2, i3, 4]
        x5 += x[i1, i2, i3, 5]
        x6 += x[i1, i2, i3, 6]
    end
    return [x1, x2, x3, x4, x5, x6] / (N1 * N2 * N3)
end


function eq_err(sig::Array{<:Number,4}, args...)
    err = 0.0
    N1, N2, N3, _ = size(sig)
    @inbounds for k in 1:N3, j in 1:N2, i in 1:N1
        err += sig[i, j, k, 1]^2 + sig[i, j, k, 2]^2 + sig[i, j, k, 3]^2 + 2 * sig[i, j, k, 4]^2 + 2 * sig[i, j, k, 5]^2 + 2 * sig[i, j, k, 6]^2
    end
    return abs(err) / (N1 * N2 * N3)
end



function eq_error!(r, sig, cartesian, nelmt)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= nelmt

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        r[i] = sig[i1, i2, i3, 1] * sig[i1, i2, i3, 1] + sig[i1, i2, i3, 2] * sig[i1, i2, i3, 2] + sig[i1, i2, i3, 3] * sig[i1, i2, i3, 3] + 2 * sig[i1, i2, i3, 4] * sig[i1, i2, i3, 4] + 2 * sig[i1, i2, i3, 5] * sig[i1, i2, i3, 5] + 2 * sig[i1, i2, i3, 6] * sig[i1, i2, i3, 6]

    end
    return nothing
end


function eq_err(sig::CuArray, cartesian, r)
    nelmt = size(sig, 1) * size(sig, 2) * size(sig, 3)
    n_blocks, n_threads = get_blocks_threads(nelmt)
    @cuda blocks = n_blocks threads = n_threads eq_error!(r, sig, cartesian, nelmt)
    residu = reduce(+, r)
    residu = abs(residu) / nelmt
end




function meanfield(x::CuArray{<:Number,4})

    Nlength = size(x, 1) * size(x, 2) * size(x, 3)

    sums = CUDA.sum(reshape(x, :, size(x, 4)), dims=1)
    X = vec(sums) ./ Nlength
    return Array(X)
end
