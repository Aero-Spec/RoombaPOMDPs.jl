using POMDPs
using StaticArrays
using Random
using Distributions
import ParticleFilters

# Utility: 2D vector for action noise
const SVec2 = SVector{2, Float64}

# Utility to generate a random RoombaState
function random_state(model, rng)
    return RoombaState(
        rand(rng, Uniform(-1.0, 1.0)),    # x
        rand(rng, Uniform(-1.0, 1.0)),    # y
        rand(rng, Uniform(-π, π)),        # th
        0.0                               # terminal flag
    )
end

# For preallocating particle memory
function particle_memory(model)
    T = typeof(random_state(model, MersenneTwister(0)))
    return T[]
end

"""
    RoombaParticleFilter

A custom particle filter for the Roomba environment.
"""
mutable struct RoombaParticleFilter{M<:RoombaModel, RM, RNG<:AbstractRNG, PMEM} <: Updater
    model::M
    resampler::RM
    n_init::Int
    v_noise_coeff::Float64
    om_noise_coeff::Float64
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

function RoombaParticleFilter(
    model, n::Integer, v_noise_coeff, om_noise_coeff;
    resampler=ParticleFilters.LowVarianceResampler(n),
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    return RoombaParticleFilter(
        model,
        resampler,
        n,
        v_noise_coeff,
        om_noise_coeff,
        rng,
        sizehint!(particle_memory(model), n),
        sizehint!(Float64[], n)
    )
end

"""
    POMDPs.update(up::RoombaParticleFilter, b::ParticleFilters.ParticleCollection, a, o)

Particle filter update for Roomba. Propagates particles, adds action noise, computes observation weights, and resamples.
"""
function POMDPs.update(up::RoombaParticleFilter, b::ParticleFilters.ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)
    all_terminal = true

    for s in ParticleFilters.particles(b)
        if !isterminal(up.model, s)
            all_terminal = false

            a_pert = a + SVec2(
                up.v_noise_coeff * (rand(up.rng) - 0.5),
                up.om_noise_coeff * (rand(up.rng) - 0.5)
            )

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
        up.model,    # model for transition (can be same as observation model)
        up.model,    # model for observation
        b, a, o,
        up.rng
    )
end

# Belief initialization: create n_init random states
function ParticleFilters.initialize_belief(up::RoombaParticleFilter, d)
    ParticleFilters.ParticleCollection([random_state(d, up.rng) for _ in 1:up.n_init])
end
