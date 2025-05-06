
export homogenize

function homogenize(
    phases::Array{<:Integer,3},
    material_list::Vector{<:Elastic},
    loading_type::LoadingType,
    loading_list::Vector{<:AbstractVector{<:Number}},
    tols::Vector{<:Real};
    verbose_fft::Bool=false,
    verbose_step::Bool=false,
    c0::Union{Nothing,<:Elastic}=nothing,
    Nit_max::Int64=1000,
    keep_fields::Bool=false,
    precision::Precision=Simple,
    gpu::Bool=true
)



    if isnothing(c0)
        c0 = choose_c0(material_list, FixedPoint, false)
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
        eps = CUDA.zeros(T, size(phases)..., 6)
        sig = CUDA.zeros(T, size(phases)..., 6)

        material_list = [IE2ITE(m) |> cu for m in material_list] |> cu
        r = CUDA.zeros(T, size(phases))
    else
        eps = zeros(T, size(phases)..., 6)
        sig = zeros(T, size(phases)..., 6)
        material_list = [IE2ITE(m) for m in material_list]
        r = nothing
    end

    EPS = meanfield(eps)
    SIG = meanfield(sig)

    P, Pinv, xi1, xi2, xi3, tau = initFFT(eps)

    cartesian = CartesianIndices(size(phases))


    step_hist = Hist{T}(length(loading_list))

    if keep_fields
        epsf = zeros(T, size(phases)..., 6, length(loading_list))
        sigf = zeros(T, size(phases)..., 6, length(loading_list))
    end


    for loading_index in eachindex(loading_list)
        loading = loading_list[loading_index]


        it, err_equi, err_load = fixed_point_step_solver!(r, eps, sig, EPS, SIG, phases, material_list, tols, loading_type, loading, c0, P, Pinv, xi1, xi2, xi3, tau, Nit_max, verbose_fft, cartesian)

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
        :steps => convert_hist(step_hist),
        :eps => keep_fields ? epsf : nothing,
        :sig => keep_fields ? sigf : nothing,
    )

    return output
end




function fixed_point_step_solver!(r, eps, sig, EPS, SIG, phases, material_list, tols::Vector, loading_type::LoadingType, loading::Vector, c0, P::AbstractFFTs.Plan, Pinv::AbstractFFTs.Plan, xi1, xi2, xi3, tau, Nit_max::Integer, verbose_fft::Bool, cartesian)


    timer_fft = 0.0
    timer_gamma0 = 0.0
    timer_ifft = 0.0
    timer_fftgamma0ifft = 0.0
    timer_update_eps = 0.0
    timer_sig = 0.0
    timer_err = 0.0
    timer_mean = 0.0


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

            tfftgamma0ifft = CUDA.@elapsed tfft, tgamma0, tifft = gamma0!(P, Pinv, xi1, xi2, xi3, tau, sig, c0, new_mean_eps)

            tupdate_eps = CUDA.@elapsed (eps isa CuArray) ? (CUDA.@. eps .+= sig) : (eps .+= sig)

            terr = CUDA.@elapsed err_equi = eq_err(sig, cartesian, r)

            tsig = CUDA.@elapsed rdc!(sig, eps, phases, material_list, cartesian)

            tmean = CUDA.@elapsed begin
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


            timer_fft += tfft
            timer_gamma0 += tgamma0
            timer_ifft += tifft

            timer_fftgamma0ifft += tfftgamma0ifft
            timer_update_eps += tupdate_eps
            timer_sig += tsig
            timer_err += terr
            timer_mean += tmean

        
        end
    end

    if verbose_fft
        println("")
        println("Total time $tit")
        println("\tGreen operator Γ⁰\t $timer_fftgamma0ifft")
        println("\t\tFFT\t\t\t $timer_fft")
        println("\t\tΓ̂⁰\t\t\t $timer_gamma0")
        println("\t\tiFFT\t\t\t $timer_ifft")
        println("\tUpdate strain ϵ=ϵ-Γ̂⁰σ\t $timer_update_eps")
        println("\tConstitutive eq. σ=f(ϵ)\t $timer_sig")
        println("\tError ‖Γ⁰σ‖\t\t $timer_err")
        println("\tCompute E=<ϵ> Σ=<σ>\t $timer_mean")
    end


    return it, err_equi, err_load

end
