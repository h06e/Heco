using Heco

function main()

    matrix = IE(E=4.0, nu=0.4)
    fibre = ITE(El=230.0, Et=15.0, nul=0.2, mul=15.0, mut=10.0)
    material_list = [matrix, fibre]

    nf=200
    f=0.5
    dmin=0.1
    Np=128

    info, micro = Micro.gen_2d_random_disks(nf, f, dmin, Np, seed=123)

    loading_list = [[1.0,0.,0.,0.,0.,0.]]
    time_list = [Float64(i) for i in eachindex(loading_list)]

    tols = [1e-6, 1e-4]


    sol = homogenize(
        micro,
        material_list,
        Strain,
        loading_list,
        time_list,
        tols;
        verbose_fft=true,
        verbose_step=true,
        Nit_max=500,
        precision=Simple,
        c0=nothing,
        gpu=false
    )


end

main()