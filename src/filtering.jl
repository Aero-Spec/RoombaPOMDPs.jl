using Random
using StaticArrays
using ParticleFilters: LowVarianceResampler, WeightedParticleBelief, ParticleCollection, particles
using POMDPs

# 2D vector type for action noise (v, ω)
const SVec2 = SVector{2, Float64}

# Allocate a correctly-typed particle buffer for this model.
# We try to infer the concrete state type from the model; otherwise fall back to RoombaState.
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

# Main constructor (positional API preserved). `resampler` is optional; we build a sensible default.
function RoombaParticleFilter(
    model,
    n::Integer,
    v_noise_coeff,
    om_noise_coeff,
    resampler::Union{Nothing,Any}=nothing,
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    rm = resampler === nothing ? (try
            LowVarianceResampler(n)
        catch
            LowVarianceResampler()
        end) : resampler

    return RoombaParticleFilter(
        model,
        rm,
        n,
        v_noise_coeff,
        om_noise_coeff,
        rng,
        sizehint!(particle_memory(model), n),
        sizehint!(Float64[], n),
    )
end

# Belief update with action noise injected (keeps your original logic)
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

    # Resample a new ParticleCollection from weighted particles
    return ParticleFilters.resample(
        up.resampler,
        WeightedParticleBelief(pm, wm),
        up.model, up.model,  # keep signatures compatible with ParticleFilters versions that expect these
        b, a, o,
        up.rng,
    )
end

# Initialize belief with n_init prior samples from a distribution d
ParticleFilters.initialize_belief(up::RoombaParticleFilter, d) =
    ParticleCollection([rand(up.rng, d) for _ in 1:up.n_init])
