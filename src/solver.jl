
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

        # if save_fields
        #     #todo save to vtk or any
        #     @info "save_fields option not implemented yet"
        # end


    end

    output = Dict(
        :steps => step_hist,
        :eps => keep_fields ? epsf : nothing,
        :sig => keep_fields ? sigf : nothing,
    )

    # return output
end




function fixed_point_step_solver!(r, eps, sig, EPS, SIG, phases, material_list, tols::Vector, loading_type::LoadingType, loading::Vector, c0, P::AbstractFFTs.Plan, Pinv::AbstractFFTs.Plan, xi1, xi2, xi3, tau, Nit_max::Integer, verbose_fft::Bool, cartesian)

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

    while (err_equi > tol_equi || err_load > tol_load) && it < Nit_max
        it += 1

        if loading_type == Strain
            new_mean_eps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else
            new_mean_eps = compute_eps(loading - SIG, c0)
        end

        gamma0!(P, Pinv, xi1, xi2, xi3, tau, sig, c0, new_mean_eps)

        (eps isa CuArray) ? (CUDA.@. eps .+= sig) : (eps .+= sig)

        err_equi = eq_err(sig, cartesian, r)

        rdc!(sig, eps, phases, material_list, cartesian)

        EPS .= meanfield(eps)
        SIG .= meanfield(sig)

        if loading_type == Strain
            err_load = 0.0
        else
            err_load = abs(sum((SIG .- loading) .^ 2 .* [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))
        end

        verbose_fft ? print_iteration(it, EPS, SIG, err_equi, err_load, tols) : nothing
        isnan(err_equi) ? (@error "Error equals NaN -> Divergence (bad choice for c0)") : nothing
    end

    return it, err_equi, err_load

end
