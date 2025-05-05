

function meanfield(x::Array{<:Number,4})
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    x5 = 0.0
    x6 = 0.0

    _, N1, N2, N3 = size(x)
    @inbounds begin
        for i3 in 1:N3
            for i2 in 1:N2
                for i1 in 1:N1
                    x1 += x[1, i1, i2, i3]
                    x2 += x[2, i1, i2, i3]
                    x3 += x[3, i1, i2, i3]
                    x4 += x[4, i1, i2, i3]
                    x5 += x[5, i1, i2, i3]
                    x6 += x[6, i1, i2, i3]
                end
            end
        end
    end
    return [x1, x2, x3, x4, x5, x6] / (N1 * N2 * N3)
end


function eq_err(sig::Array{<:Number,4}, args...)
    err = 0.0
    _, N1, N2, N3 = size(sig)
    @inbounds begin
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1
                    err += sig[1, i, j, k]^2 + sig[2, i, j, k]^2 + sig[3, i, j, k]^2 + 2 * sig[4, i, j, k]^2 + 2 * sig[5, i, j, k]^2 + 2 * sig[6, i, j, k]^2
                end
            end
        end
    end
    return abs(err) / (N1 * N2 * N3)
end



function eq_error!(r, sig, cartesian, nelmt)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= nelmt

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        r[i] = sig[1, i1, i2, i3] * sig[1, i1, i2, i3] + sig[2, i1, i2, i3] * sig[2, i1, i2, i3] + sig[3, i1, i2, i3] * sig[3, i1, i2, i3] + 2 * sig[4, i1, i2, i3] * sig[4, i1, i2, i3] + 2 * sig[5, i1, i2, i3] * sig[5, i1, i2, i3] + 2 * sig[6, i1, i2, i3] * sig[6, i1, i2, i3]
        # r[i] = sig1[i]*sig1[i] + sig2[i]*sig2[i] + sig3[i]*sig3[i] + 2 * sig4[i]*sig4[i] + 2 * sig5[i]*sig5[i] + 2 * sig6[i]*sig6[i]
    end
    return nothing
end


function eq_err(sig::CuArray, cartesian, r)
    nelmt = size(sig, 2) * size(sig, 3) * size(sig, 4)
    n_blocks, n_threads = get_blocks_threads(nelmt)
    @cuda blocks = n_blocks threads = n_threads eq_error!(r, sig, cartesian, nelmt)
    residu = reduce(+, r)
    residu = abs(residu) / nelmt
end




function meanfield(x)

    Nlength = size(x, 2) * size(x, 3) * size(x, 4)

    sums = CUDA.sum(reshape(x, size(x, 1), :), dims=2)
    X = vec(sums) ./ Nlength
    return Array(X)
end
