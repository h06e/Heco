
export homogenize

function homogenize(
    phases::Array{<:Number,3},
    material_list::Vector{<:Elastic},
    loading_type::LoadingType,
    loading_list::Vector{<:AbstractVector{<:Number}},
    time_list::Vector{<:Real},
    tols::Vector{<:Real};
    keep_it_info::Bool=false,
    verbose_fft::Bool=false,
    verbose_step::Bool=false,
    c0::Union{Nothing,<:Elastic}=nothing,
    Nit_max::Int64=1000,
    scheme::Scheme=FixedPoint,
    polarization_AB::Vector{Float64}=[2.0, 2.0],
    polarization_skip_tests::Int64=0,
    keep_fields::Bool=false,
    save_fields::Bool=false,
    precision::Precision=Simple,
    gpu::Bool=true
)



    if isnothing(c0)
        c0 = choose_c0(material_list, scheme, false)
    end
    verbose_step ? (@info "c0" c0) : nothing

    flag_complex = any(x -> x isa IE{ComplexF32} || x isa IE{ComplexF64}, material_list)

    c0 = convert_precision(c0, precision)
    material_list = convert_precision(material_list, precision)

    if precision == Simple
        flag_complex ? T = ComplexF32 : T = Float32
        FT = ComplexF32

    else
        flag_complex ? T = ComplexF64 : T = Float64
        FT = ComplexF64
    end

    phases = map(Int32, phases)

    if gpu
        phases = cu(phases)
        eps = CUDA.zeros(T, 6, size(phases)...)
        sig = CUDA.zeros(T, 6, size(phases)...)

        material_list = [IE2ITE(m) |> cu for m in material_list] |> cu
        r = CUDA.zeros(T, size(phases))
    else
        eps = zeros(T, 6, size(phases)...)
        sig = zeros(T, 6, size(phases)...)
        r = nothing
    end

    EPS = meanfield(eps)
    SIG = meanfield(sig)

    P, Pinv, xi1, xi2, xi3, tau = initFFT(eps)

    cartesian = CartesianIndices(size(phases))

    if scheme == FixedPoint
        step_solver! = fixed_point_step_solver!
    else
        #Todo 
        @error "Polarization scheme not implemented yet."
        return
    end

    step_hist = Hist{T}(length(loading_list))

    if keep_fields
        epsf = zeros(FT, 6, size(phases)..., length(loading_list))
        sigf = zeros(FT, 6, size(phases)..., length(loading_list))
    end


    for loading_index in eachindex(loading_list)
        loading = loading_list[loading_index]


        it, err_equi, err_load = step_solver!(r, eps, sig, EPS, SIG, phases, material_list, tols, loading_type, loading, c0, P, Pinv, xi1, xi2, xi3, tau, Nit_max, verbose_fft, cartesian)

        verbose_step ? print_iteration(it, EPS, SIG, err_equi, err_load, tols) : nothing
        (it == Nit_max) ? (@error "MAX ITERATIONS REACHED" return) : nothing


        ES = sum([1.0, 1.0, 1.0, 2.0, 2.0, 2.0] .* EPS .* SIG)
        update_hist!(step_hist, loading_index, EPS, SIG, ES, err_equi, err_load, it)


        if keep_fields
            epsf[:, :, :, :, loading_index] .= Array(eps)
            sigf[:, :, :, :, loading_index] .= Array(sig)
        end
    end
    
 
    output = Dict(
        :steps => step_hist,
        :eps => keep_fields ? epsf : nothing,
        :sig => keep_fields ? sigf : nothing,
    )

    return output
end




function fixed_point_step_solver!(r, eps, sig, EPS, SIG, phases, material_list, tols::Vector, loading_type::LoadingType, loading::Vector, c0, P::AbstractFFTs.Plan, Pinv::AbstractFFTs.Plan, xi1, xi2, xi3, tau, Nit_max::Integer, verbose_fft::Bool, cartesian)


    chrono_tfft1 = 0.0
    chrono_tgammafft = 0.0
    chrono_tfft2 = 0.0

    chrono_gamma0 = 0.0
    chrono_majeps = 0.0
    chrono_sig = 0.0
    chrono_err = 0.0
    chrono_mean = 0.0


    if loading_type == Strain
        add_mean_value!(eps, loading .- EPS, cartesian)
    else
        new_mean_eps = compute_eps(loading .- EPS, c0)
        add_mean_value!(eps, new_mean_eps, cartesian)

    end
    rdc!(sig, eps, phases, material_list, cartesian)

    EPS .= meanfield(eps)
    SIG .= meanfield(sig)

    tol_equi = tols[1]
    tol_load = tols[2]
    err_equi = 1e9
    err_load = 1e9
    it = 0

    tit = @elapsed begin
    while (err_equi > tol_equi || err_load > tol_load) && it < Nit_max
        it += 1

        if loading_type == Strain
            new_mean_eps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else
            new_mean_eps = compute_eps(loading - SIG, c0)
        end

        t_gamma0 = CUDA.@elapsed tfft1, tgammafft, tfft2 = gamma0!(P, Pinv, xi1, xi2, xi3, tau, sig, c0, new_mean_eps)

        t_majeps = CUDA.@elapsed (eps isa CuArray) ? (CUDA.@. eps .+= sig) : (eps .+= sig)

        t_equi = CUDA.@elapsed err_equi = eq_err(sig, cartesian, r)

        t_rdc = CUDA.@elapsed rdc!(sig, eps, phases, material_list, cartesian)

        t_mean = CUDA.@elapsed begin
        EPS .= meanfield(eps)
        SIG .= meanfield(sig)
        end

        if loading_type == Strain
            err_load = 0.0
        else
            err_load = abs(sum((SIG .- loading) .^ 2 .* [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))
        end

        verbose_fft ? print_iteration(it, EPS, SIG, err_equi, err_load, tols) : nothing
        isnan(err_equi) ? (@error "Error equals NaN -> Divergence (bad choice for c0)") : nothing


        chrono_tfft1 += tfft1
        chrono_tgammafft += tgammafft
        chrono_tfft2 += tfft2

        chrono_gamma0 += t_gamma0
        chrono_majeps += t_majeps
        chrono_sig += t_rdc
        chrono_err += t_equi
        chrono_mean += t_mean
    end
    end
    println("")
    println("Temps total $tit")
    println("")
    println("chrono_tfft1  = $chrono_tfft1")
    println("chrono_tgammafft = $chrono_tgammafft")
    println("chrono_tfft2 = $chrono_tfft2")
    println("")
    println("chrono_gamma0 (+ fft + ifft) = $chrono_gamma0")
    println("chrono_majeps = $chrono_majeps")
    println("chrono_sig0 = $chrono_sig")
    println("chrono_err = $chrono_err")
    println("chrono_mean = $chrono_mean")


    return it, err_equi, err_load

end
