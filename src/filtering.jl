using ParticleFilters: LowVarianceResampler, WeightedParticleBelief

"""
Definition of the particle filter for the Roomba environment
Fields:
- `v_noise_coeff::Float64` coefficient to scale particle-propagation noise in velocity
- `om_noise_coeff::Float64` coefficient to scale particle-propagation noise in turn-rate
"""

function particle_memory(model)
    Vector{RoombaState}()
end

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

function RoombaParticleFilter(
    model, n::Integer, v_noise_coeff, om_noise_coeff, 
    resampler=LowVarianceResampler(n), rng::AbstractRNG=Random.GLOBAL_RNG
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

function POMDPs.update(up::RoombaParticleFilter, b::Vector{RoombaState}, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)
    all_terminal = true
    for s in b
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

    return WeightedParticleBelief(pm, wm)
end

ParticleFilters.initialize_belief(up::RoombaParticleFilter, d) = [rand(up.rng, d) for i in 1:up.n_init]

# --- POLICY/CONTROL FALLBACKS ---

struct ToEnd end

function POMDPTools.action(p::ToEnd, s::RoombaState)
    # TODO: Replace with actual logic for your robot
    return RoombaAct(0.0, 0.0)
end

function POMDPTools.action(p::ToEnd, b::WeightedParticleBelief{RoombaState})
    idx = argmax(b.weights)
    s = b.particles[idx]
    return POMDPTools.action(p, s)
end
