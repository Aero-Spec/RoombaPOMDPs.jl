using POMDPs
using StaticArrays
using Random
using Distributions
import ParticleFilters

const SVec2 = SVector{2, Float64}

function random_state(model, rng)
    return RoombaState(
        rand(rng, Distributions.Uniform(-1.0, 1.0)),   # x
        rand(rng, Distributions.Uniform(-1.0, 1.0)),   # y
        rand(rng, Distributions.Uniform(-π, π)),       # th
        0.0                                            # terminal flag
    )
end


# THIS MUST BE DEFINED BEFORE RoombaParticleFilter
function particle_memory(model)
    T = typeof(random_state(model, MersenneTwister(0)))
    return T[]
end

"""
Definition of the particle filter for the Roomba environment
Fields:
- `v_noise_coeff::Float64` coefficient to scale particle-propagation noise in velocity
- `om_noise_coeff::Float64` coefficient to scale particle-propagation noise in turn-rate
"""
mutable struct RoombaParticleFilter{M<:RoombaModel,RM,RNG<:AbstractRNG,PMEM} <: Updater
    model::M
    resampler::RM
    n_init::Int
    v_noise_coeff::Float64
    om_noise_coeff::Float64
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

function RoombaParticleFilter(model, n::Integer, v_noise_coeff, om_noise_coeff, resampler=ParticleFilters.LowVarianceResampler(n), rng::AbstractRNG=Random.GLOBAL_RNG)
    return RoombaParticleFilter(model,
                               resampler,
                               n,
                               v_noise_coeff,
                               om_noise_coeff,
                               rng,
                               sizehint!(particle_memory(model), n),
                               sizehint!(Float64[], n)
                              )
end

function POMDPs.update(up::RoombaParticleFilter, b::ParticleFilters.ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)
    all_terminal = true
    for s in ParticleFilters.particles(b)
        if !isterminal(up.model, s)
            all_terminal = false
            a_pert = a + SVec2(up.v_noise_coeff * (rand(up.rng) - 0.5), up.om_noise_coeff * (rand(up.rng) - 0.5))
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            push!(pm, sp)
            push!(wm, obs_weight(up.model, s, a_pert, sp, o))
        end
    end
    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
    end

    return ParticleFilters.resample(
        up.resampler,
        ParticleFilters.WeightedParticleBelief(pm, wm, sum(wm), nothing),
        up.model,
        up.model,
        b, a, o,
        up.rng
    )
end

# initialize belief state
function ParticleFilters.initialize_belief(up::RoombaParticleFilter, d)
    ParticleFilters.ParticleCollection([random_state(d, up.rng) for i in 1:up.n_init])
end
