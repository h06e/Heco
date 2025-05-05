module Micro
using StaticArrays
import Random

export gen_2d_random_disks


struct Disk
    x::SVector{2,Float64}  # current position of the disk
    xp::SVector{2,Float64} # previous position of the disk
    r::Float64
end

function random_new_disk(initial_speed::Float64)
    x = SVector{2,Float64}(rand(), rand())
    xp = x + initial_speed * (rand(2) .- 0.5)
    r = 0.0
    return Disk(x, xp, r)
end

function update_inertia!(disk_list::Vector{Disk}, damping::Float64)
    for is in eachindex(disk_list)
        s = disk_list[is]
        u = s.x - s.xp
        x_new = s.x + u * (1 - damping)
        xp = s.x
        updated_disk = Disk(x_new, xp, s.r)
        disk_list[is] = updated_disk
    end
end

function update_periodicity!(disk_list::Vector{Disk})
    for is in eachindex(disk_list)
        s = disk_list[is]
        xx = s.x[1]
        if xx < 0
            xx_new = s.x[1] + 1
            xxp_new = s.xp[1] + 1
        elseif xx > 1
            xx_new = s.x[1] - 1
            xxp_new = s.xp[1] - 1
        else
            xx_new = s.x[1]
            xxp_new = s.xp[1]
        end

        xy = s.x[2]
        if xy < 0
            xy_new = s.x[2] + 1
            xyp_new = s.xp[2] + 1
        elseif xy > 1
            xy_new = s.x[2] - 1
            xyp_new = s.xp[2] - 1
        else
            xy_new = s.x[2]
            xyp_new = s.xp[2]
        end

        disk_list[is] = Disk(SVector{2,Float64}(xx_new, xy_new), SVector{2,Float64}(xxp_new, xyp_new), s.r)
    end
end

function update_contact!(disk_list::Vector{Disk}, stiffness::Float64)
    d_min_meas = 1e9
    
    for is1 in eachindex(disk_list)
        for is2 in eachindex(disk_list)
            if is1 != is2
                s1 = disk_list[is1]
                s2 = disk_list[is2]

                k1 = round(s1.x[1] - s2.x[1])
                k2 = round(s1.x[2] - s2.x[2])

                closestB2A_dist2 = sum((s1.x - s2.x .- [k1,k2]) .^2)
                d = sqrt(closestB2A_dist2)
                d_min_meas = min(d_min_meas, d)

                if closestB2A_dist2 <= (2 * s1.r)^2
                    normal = s2.x + [k1,k2] - s1.x
                    overlap = 2 * s1.r - d
                    disp = normal * stiffness / d * 0.5 * overlap

                    x_s1_new = s1.x - disp
                    x_s2_new = s2.x + disp

                    disk_list[is1] = Disk(x_s1_new, s1.xp, s1.r)
                    disk_list[is2] = Disk(x_s2_new, s2.xp, s2.r)
                end
            end
        end
    end
    return d_min_meas
end

function update_radius!(disk_list::Vector{Disk}, radius::Float64)
    for is in eachindex(disk_list)
        s = disk_list[is]
        disk_list[is] = Disk(s.x, s.xp, radius)
    end
end


function add_sphere!(img,cx,cy,r)
    Np = size(img,1)
    xm, xM = Int(floor(max(1,cx*Np-r*Np))), Int(ceil(min(Np,cx*Np+r*Np)))
    ym, yM = Int(floor(max(1,cy*Np-r*Np))), Int(ceil(min(Np,cy*Np+r*Np)))
    for i in xm:xM
        for j in ym:yM
            d2=(i/Np-cx)^2+(j/Np-cy)^2
            if d2 <= (r^2)
                @inbounds img[i,j] = 1
            end
        end
    end
    return img
end

function conv2array(disk_list::Vector{Disk}, Np::Int64)
    img = zeros(Np, Np)

    for s in disk_list
        cx, cy = s.x
        
        img = add_sphere!(img,cx,cy,s.r)
        if cx < s.r
            if cy < s.r
                img = add_sphere!(img,cx+1,cy,s.r)
                img = add_sphere!(img,cx+1,cy+1,s.r)
                img = add_sphere!(img,cx,cy+1,s.r)
            elseif cy > 1 - s.r
                img = add_sphere!(img,cx+1,cy,s.r)
                img = add_sphere!(img,cx+1,cy-1,s.r)
                img = add_sphere!(img,cx,cy-1,s.r)
            else
                img = add_sphere!(img,cx+1,cy,s.r)
            end

        elseif cx > 1 - s.r
            if cy < s.r
                img = add_sphere!(img,cx-1,cy,s.r)
                img = add_sphere!(img,cx-1,cy+1,s.r)
                img = add_sphere!(img,cx,cy+1,s.r)
            elseif cy > 1 - s.r
                img = add_sphere!(img,cx-1,cy,s.r)
                img = add_sphere!(img,cx-1,cy-1,s.r)
                img = add_sphere!(img,cx,cy-1,s.r)
            else
                img = add_sphere!(img,cx-1,cy,s.r)
            end
        else
            if cy < s.r
                img = add_sphere!(img,cx,cy+1,s.r)
            elseif cy > 1 - s.r
                img = add_sphere!(img,cx,cy-1,s.r)
            end
        end

    end
    return img
end




"""
Function that generate a 2D microstructure with random stacking disks.
Inputs:
- nf: the number of disks
- f: the volume fraction of disks
- d_min: the minimal distance bewteen two disks (x disk radius)
- Np: the returned discretization
other inputs
- seed: specify seed for reproductibility
- damping: damping slowing the particules
- stiffness: stiffness of the contact during collision
- initial_speed: initial velocities of the particles at the fist step
- growing_rate: number of steps within the particles will grow
- it_max: maximal number of iteration to solve the contact
- tol: tolerance distance of the contact
- verbose: print simulation data

Outputs
- info: dictonnary that contain data about the microstructure
- img: final microstructure
"""
function gen_2d_random_disks(nf::Int64, f::Float64, d_min::Float64, Np::Int64;
    damping=0.1::Float64,
    stiffness=1.0::Float64,
    initial_speed=0.2::Float64,
    growing_rate=50::Int64,
    it_max=200::Int64,
    tol=1e-4::Float64,
    verbose=false::Bool,
    seed=nothing::Union{Nothing,Int64})

    isnothing(seed) ? (seed=rand(Int64)) : nothing
    Random.seed!(seed)
    
    if f < 0 || f > 0.92
        @error "f must be between 0 and 0.92"
        throw(ArgumentError)
    end

    if d_min < 0
        @error "d_min must be >= 0"
        throw(ArgumentError)
    end

    true_radius_radius = sqrt(f / (nf * π)) # pix in radius
    final_radius = true_radius_radius * (1 + d_min / 2)

    disk_list = [random_new_disk(initial_speed) for _ in 1:nf]

    #! Fisrt phase : growing phase
    i = 1
    for r2 in LinRange(0, final_radius^2, growing_rate) #sqared root growing
        update_radius!(disk_list, sqrt(r2))
        update_periodicity!(disk_list)
        d_min_meas = update_contact!(disk_list, stiffness)
        update_inertia!(disk_list, damping)
        verbose ? (@info "$i d_min_meas = $d_min_meas") : nothing
        i += 1
    end

    #! Second phase : waiting for the contact to be "enough" solved
    d_min_meas = 0.0
    while d_min_meas < 2 * final_radius - tol && i < it_max
        update_periodicity!(disk_list)
        d_min_meas = update_contact!(disk_list, stiffness)
        update_inertia!(disk_list, damping)
        verbose ? (@info "$i d_min_meas = $d_min_meas | dmin-tol =$(2*final_radius-tol)") : nothing
        i += 1
    end
    update_periodicity!(disk_list)

    #! Apply the true radius to take d_min into account
    update_radius!(disk_list, true_radius_radius)
    img = conv2array(disk_list, Np)
    img = ones(size(img)) .* (img .> 0)

    #! Compute the true fibre fraction
    true_f = sum(img)./length(img)
    true_d_min = d_min_meas / true_radius_radius - 2

    #! Numbering the phases bewteen 1 and  inf
    img = reshape(img, size(img, 1), size(img, 2), 1)
    img .+= 1 # to match with material_list indexes
    img = map(Int32, img)

    info = Dict(
        :f =>true_f,
        :dmin => true_d_min,
        :Dfpix => Int(round(2*true_radius_radius*Np)),
    )

    return info, img
end


end
