# specification of particle filters for the bumper and lidar Roomba environments
# maintained by {jmorton2,kmenda}@stanford.edu

using ParticleFilters: LowVarianceResampler, ParticleCollection, WeightedParticleBelief
using Random
using StaticArrays

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

function RoombaParticleFilter(model, n::Integer, v_noise_coeff, om_noise_coeff, resampler=LowVarianceResampler(n), rng::AbstractRNG=Random.GLOBAL_RNG)
    pmem = sizehint!(Vector{typeof(rand(rng, model))}(), n)
    wmem = sizehint!(Float64[], n)
    return RoombaParticleFilter(model, resampler, n, v_noise_coeff, om_noise_coeff, rng, pmem, wmem)
end

# Modified Update function adds noise to the actions that propagate particles
function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)
    all_terminal = true
    for s in particles(b)
        if !isterminal(up.model, s)
            all_terminal = false
            # noise added here:
            a_pert = a + SVec2(up.v_noise_coeff * (rand(up.rng) - 0.5), up.om_noise_coeff * (rand(up.rng) - 0.5))
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            push!(pm, sp)
            push!(wm, obs_weight(up.model, s, a_pert, sp, o))
        end
    end
    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
    end

    return ParticleFilters.resample(  # fully qualified
        up.resampler,
        WeightedParticleBelief(pm, wm, sum(wm), nothing),
        up.model,
        up.model,
        b, a, o,
        up.rng
    )
end

# Initialize belief state (extension of POMDPs.initialize_belief)
function initialize_belief(up::RoombaParticleFilter, d)
    return ParticleCollection([rand(up.rng, d) for _ in 1:up.n_init])
end
