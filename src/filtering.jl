using POMDPs
using StaticArrays
using Random
using Distributions
using ParticleFilters: particles, ParticleCollection, LowVarianceResampler
import ParticleFilters: initialize_belief

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

# ---------- local resampling (no ParticleFilters.resample needed) ----------
# Normalize weights in place; if invalid, make them uniform
function _normalize!(w::Vector{Float64})
    s = sum(w)
    if !(isfinite(s)) || s <= 0
        fill!(w, 1.0 / length(w))
    else
        invs = 1.0 / s
        @inbounds for i in eachindex(w)
            w[i] *= invs
        end
    end
    return w
end

# Low-variance (systematic) resampling: return indices of chosen parents
function _lv_indices(w::Vector{Float64}, n::Int, rng::AbstractRNG)
    _normalize!(w)
    idx = Vector{Int}(undef, n)
    u0 = rand(rng) / n
    c = w[1]
    i = 1
    @inbounds for m in 1:n
        u = u0 + (m-1)/n
        while u > c && i < length(w)
            i += 1
            c += w[i]
        end
        idx[m] = i
    end
    return idx
end

# Build a new ParticleCollection by resampling pm with weights wm to size n_out
function _resample_particles(pm::Vector, wm::Vector{Float64}, n_out::Int, rng::AbstractRNG)
    inds = _lv_indices(copy(wm), n_out, rng) # copy so we don't mutate caller's wm
    res = Vector{eltype(pm)}(undef, n_out)
    @inbounds for j in 1:n_out
        res[j] = pm[inds[j]]
    end
    return ParticleCollection(res)
end
# --------------------------------------------------------------------------

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
    resampler=LowVarianceResampler(n),  # kept for API compatibility
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
    POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection, a, o)

Particle filter update for Roomba. Propagates particles, adds action noise, computes observation weights, and resamples.
"""
function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm); empty!(wm)
    all_terminal = true

    for s in particles(b)
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

    if all_terminal || isempty(pm)
        error("Particle filter update error: all states in the particle collection were terminal.")
    end

    # Resample to the original belief size
    n_out = length(particles(b))
    return _resample_particles(pm, wm, n_out, up.rng)
end

# Belief initialization: create n_init random states
function initialize_belief(up::RoombaParticleFilter, d)
    ParticleCollection([random_state(d, up.rng) for _ in 1:up.n_init])
end
