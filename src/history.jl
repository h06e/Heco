using Printf


struct Hist{T}
    E::Array{T,2}
    S::Array{T,2}
    ES::Vector{T}
    equi::Vector{Float64}
    load::Vector{Float64}
    it::Vector{Int64}

    function Hist{T}(n::Int64) where {T<:Number}
        new(zeros(T, 6, n), zeros(T, 6, n), zeros(T, n), zeros(Float64, n), zeros(Float64, n), zeros(Int64, n))
    end
end

function update_hist!(hist::Hist, i::Int64, E::Vector, S::Vector, ES::Float64, equi::Float64, load::Float64, it::Int64)

    hist.E[:, i] .= E
    hist.S[:, i] .= S
    hist.ES[i] = ES
    hist.equi[i] = equi
    hist.load[i] = load
    hist.it[i] = it

end


function print_iteration(it::Int64, E::Vector{<:Real}, S::Vector{<:Real}, equi::Real, load::Real, tols::Vector{<:Real})
    printstyled("It: $it ")

    s = @sprintf("EQ:% 1.1e ", equi)
    equi[end] > tols[1] ? c = :red : c = :green
    printstyled(s, color=c, reverse=true)

    s = @sprintf("LO:% 1.1e ", load)
    load[end] > tols[2] ? c = :red : c = :green
    printstyled(s, color=c, reverse=true)

    s = @sprintf("| E11:% 1.5e E22:% 1.5e E33:% 1.5e E23:% 1.5e E13:% 1.5e E12:% 1.5e |",
        E...)
    printstyled(s, color=:light_blue, reverse=true)

    s = @sprintf(" S11:% 1.5e S22:% 1.5e S33:% 1.5e S23:% 1.5e S13:% 1.5e S12:% 1.5e\n",
        S...)
    printstyled(s, color=:light_blue, reverse=false)
end

function print_iteration(it::Int64, E::Vector{<:Complex}, S::Vector{<:Complex}, equi::Real, load::Real, tols::Vector{<:Real})
    printstyled("It: $it ")

    s = @sprintf("EQ:% 1.1e ", equi)
    equi[end] > tols[1] ? c = :red : c = :green
    printstyled(s, color=c, reverse=true)

    s = @sprintf("LO:% 1.1e ", load)
    load[end] > tols[2] ? c = :red : c = :green
    printstyled(s, color=c, reverse=true)

    s = @sprintf("| E11:% 1.5e + % 1.5ei E22:% 1.5e + % 1.5ei E33:% 1.5e + % 1.5ei E23:% 1.5e + % 1.5ei E13:% 1.5e + % 1.5ei E12:% 1.5e + % 1.5ei |",
        collect(Iterators.flatten(zip(real.(E), imag.(E))))...)
    printstyled(s, color=:light_blue, reverse=true)

    s = @sprintf(" S11:% 1.5e + % 1.5ei S22:% 1.5e + % 1.5ei S33:% 1.5e + % 1.5ei S23:% 1.5e + % 1.5ei S13:% 1.5e + % 1.5ei S12:% 1.5e + % 1.5ei\n",
        collect(Iterators.flatten(zip(real.(S), imag.(S))))...)
    printstyled(s, color=:light_blue, reverse=false)
end