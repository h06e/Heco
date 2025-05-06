using Optim
using LinearAlgebra

export choose_c0

function getKmat(mat)
    m = IE2ITE(mat)
    return [
        [m.k + m.m m.k - m.m m.l 0.0 0.0 0.0];
        [m.k - m.m m.k + m.m m.l 0.0 0.0 0.0];
        [m.l m.l m.n 0.0 0.0 0.0];
        [0.0 0.0 0.0 2 * m.p 0.0 0.0];
        [0.0 0.0 0.0 0.0 2 * m.p 0.0];
        [0.0 0.0 0.0 0.0 0.0 2 * m.m]
    ]
end


function mat2x(mat, forceIE::Bool)
    m = IE2ITE(mat)
    return forceIE ? abs.([m.k, m.m]) : abs.([m.k, m.l, m.m, m.n, m.p])
end

function reconstruct_material(x, forceIE::Bool)
    return forceIE ? IE(kappa=x[1], mu=x[2]) : ITE(k=x[1], l=x[2], m=x[3], n=x[4], p=x[5])
end

function build_costfunc(material_list, scheme::Scheme, forceIE::Bool)
    return function (x)
        C0 = getKmat(reconstruct_material(x, forceIE))
        s = 0.0
        for m in material_list
            C = getKmat(m)
            δC = C - C0
            if scheme == FixedPoint
                s = max(s, svd(inv(C0) * δC).S[1])
            elseif scheme == Polarization
                try
                    s = max(s, abs(svd(inv(δC) * C).S[1]))
                    s = max(s, abs(svd(inv(δC) * C0).S[1]))
                catch
                    return 1e9
                end
            end
        end
        return s
    end
end

function choose_c0(material_list::Vector{<:Elastic}, scheme::Scheme, forceIE::Bool; x0::Union{Nothing,<:Elastic}=nothing)
    if material_list isa Vector{<:IE}
        kappa_list = [abs(m.kappa) for m in material_list]
        mu_list = [abs(m.mu) for m in material_list]

        k0 = scheme == FixedPoint ?
             0.5 * (minimum(kappa_list) + maximum(kappa_list)) :
             sqrt(minimum(kappa_list) * maximum(kappa_list))
        mu0 = scheme == FixedPoint ?
              0.5 * (minimum(mu_list) + maximum(mu_list)) :
              sqrt(minimum(mu_list) * maximum(mu_list))
        return IE(kappa=k0, mu=mu0)
    else

        k_av = sum(IE2ITE(mat).k for mat in material_list)/length(material_list)
        l_av = sum(IE2ITE(mat).l for mat in material_list)/length(material_list)
        m_av = sum(IE2ITE(mat).m for mat in material_list)/length(material_list)
        n_av = sum(IE2ITE(mat).n for mat in material_list)/length(material_list)
        p_av = sum(IE2ITE(mat).p for mat in material_list)/length(material_list)

        av_mat = ITE(k=k_av, l=l_av, m=m_av, n=n_av, p=p_av)

        isnothing(x0) ? x0 = mat2x(av_mat, forceIE) : nothing
        costfunc = build_costfunc(material_list, scheme, forceIE)
        result = optimize(costfunc, x0, NelderMead())
        x = Optim.minimizer(result)
        return reconstruct_material(x, forceIE)
    end
end
