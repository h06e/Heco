using Heco

function main()
    m1 = IE(E=5.0, nu=0.4)
    # m2 = IE(kappa=10.0, mu=5.0)
    m2 = ITE(El=230.0, Et=15.0, nul=0.2, mul=15.0, mut=10.0)

    N = 64
    eps = zeros(6,N,N,N)
    sig = zeros(6,N,N,N)

    phases = rand((1,2), N,N,N)

    material_list = [m1, m2]

    Heco.rdc!(sig, eps, phases, material_list)

    println(typeof(FixedPoint))
    println(typeof(material_list))

    # c0=choose_c0(material_list, FixedPoint, false)
    # println(c0)

    c0 = choose_c0(material_list, Polarization, false)
    println(c0)

end

main()