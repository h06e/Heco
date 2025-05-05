using Heco
using PyPlot

function main()
    nf=200
    f=0.5
    dmin=0.1
    Np=1024

    info, micro = Micro.gen_2d_random_disks(nf, f, dmin, Np, seed=123)

    plt.imshow(micro)
    plt.show()
end

main()