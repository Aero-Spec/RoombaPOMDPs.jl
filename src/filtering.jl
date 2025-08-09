using Random
using StaticArrays
using POMDPs
using ParticleFilters: WeightedParticleBelief, ParticleCollection, particles

# 2D vector type for action noise (v, ω)
const SVec2 = SVector{2, Float64}

# Allocate a correctly-typed particle buffer for this model.
function particle_memory(model)
    T = try
        typeof(rand(MersenneTwister(0), initialstate(model)))
    catch
        RoombaState
    end
    return T[]
end

"""
Particle filter for the Roomba environment.

Fields:
- `v_noise_coeff::Float64` : scales particle-propagation noise in velocity
- `om_noise_coeff::Float64`: scales particle-propagation noise in turn-rate
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

# Main constructor
function RoombaParticleFilter(
    model,
    n::Integer,
    v_noise_coeff,
    om_noise_coeff,
    resampler=nothing,  
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    return RoombaParticleFilter(
        model,
        resampler,
        n,
        v_noise_coeff,
        om_noise_coeff,
        rng,
        sizehint!(particle_memory(model), n),
        sizehint!(Float64[], n),
    )
end


# Works on plain arrays; returns a new vector of states.
function _systematic_resample(states::AbstractVector{T}, weights::AbstractVector{<:Real},
                              n::Integer, rng::AbstractRNG) where {T}
    n <= 0 && return T[]               # guard
    # Normalize (handle zero-sum safely)
    wsum = sum(weights)
    if !(wsum > 0)
        # all weights zero or NaN; fall back to uniform
        p = fill(1.0/n, length(states))
        return _systematic_resample(states, p, n, rng)
    end
    w = collect(weights ./ wsum)

    # cumulative
    c = cumsum(w)
    # systematic positions
    u0 = rand(rng) / n
    out = Vector{T}(undef, n)
    i = 1
    for m in 0:(n-1)
        u = u0 + m/n
        while i < length(c) && u > c[i]
            i += 1
        end
        out[m+1] = states[i]
    end
    return out
end

# Belief update with action noise injected
function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)

    all_terminal = true
    for s in particles(b)
        if !isterminal(up.model, s)
            all_terminal = false
            # add zero-mean uniform noise to action (v, ω)
            a_pert = a + SVec2(
                up.v_noise_coeff * (rand(up.rng) - 0.5),
                up.om_noise_coeff * (rand(up.rng) - 0.5),
            )
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            push!(pm, sp)
            push!(wm, obs_weight(up.model, s, a_pert, sp, o))
        end
    end

    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
    end

    # Resample locally (no ParticleFilters.resample dependency)
    new_states = _systematic_resample(pm, wm, up.n_init, up.rng)
    return ParticleCollection(new_states)
end

# Initialize belief with n_init prior samples from a distribution d
ParticleFilters.initialize_belief(up::RoombaParticleFilter, d) =
    ParticleCollection([rand(up.rng, d) for _ in 1:up.n_init])
