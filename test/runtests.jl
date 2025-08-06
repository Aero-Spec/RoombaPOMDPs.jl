using RoombaPOMDPs
using POMDPs
using POMDPTools
using ParticleFilters
using Cairo
using Gtk
using Random
using Test

# -- Environment setup --
sensor = Lidar() # or Bumper() for the bumper version of the environment
config = 3 # 1, 2, or 3
m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config));

num_particles = 2000
v_noise_coefficient = 2.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(m, num_particles, v_noise_coefficient, om_noise_coefficient)

# -- Policy definition --
mutable struct ToEnd <: Policy
    ts::Int64
end

goal_xy = get_goal_xy(m)

# action for WeightedParticleBelief
function POMDPs.action(p::ToEnd, b::WeightedParticleBelief{RoombaState})
    if p.ts < 25
        p.ts += 1
        return RoombaAct(0., 1.0)
    end
    p.ts += 1

    # go towards goal using MAP estimate
    idx = argmax(b.weights)
    s = b.particles[idx]
    goal_x, goal_y = goal_xy
    x, y, th = s[1:3]
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal - th)
    Kprop = 1.0
    om = Kprop * del_angle
    v = 5.0
    return RoombaAct(v, om)
end

# --------- FIXED: Added method for ParticleCollection -------------
function POMDPs.action(p::ToEnd, b::ParticleCollection{RoombaState})
    if p.ts < 25
        p.ts += 1
        return RoombaAct(0., 1.0)
    end
    p.ts += 1

    # Use mean of particles for action
    s = mean(b)
    goal_x, goal_y = goal_xy
    x, y, th = s[1:3]
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal - th)
    Kprop = 1.0
    om = Kprop * del_angle
    v = 5.0
    return RoombaAct(v, om)
end
# ---------------------------------------------------------------

# fallback action for Vector
function POMDPs.action(p::ToEnd, b::Vector)
    return RoombaAct(0., 1.0)
end

# fallback action for single RoombaState
function POMDPs.action(p::ToEnd, s::RoombaState)
    return RoombaAct(0., 0.)
end

# -- Simulation --
Random.seed!(0)
p = ToEnd(0)

for step in stepthrough(m, p, belief_updater, max_steps=100)
    @show step.a
end

step = first(stepthrough(m, p, belief_updater, max_steps=100))

@show fbase = tempname()
v = render(m, step)
for (ext, mime) in ["html"=>MIME("text/html"), "svg"=>MIME("image/svg+xml"), "png"=>MIME("image/png")]
    fname = fbase*"."*ext
    open(fname, "w") do f
        show(f, mime, v)
    end
    @test filesize(fname) > 0
end

# Discrete version test
m = RoombaPOMDP(
    sensor=sensor,
    mdp=RoombaMDP(
        config=config,
        aspace=vec([RoombaAct(v, om) for v in range(0, stop=2, length=2) for om in range(-2, stop=2, length=3)]),
        sspace=DiscreteRoombaStateSpace(41, 26, 20)
    )
)
@test has_consistent_initial_distribution(m)
@test has_consistent_transition_distributions(m)

belief_updater = RoombaParticleFilter(m, num_particles, v_noise_coefficient, om_noise_coefficient)

for step in stepthrough(m, RandomPolicy(m), belief_updater, max_steps=100)
    @show convert_s(RoombaState, step.s, m), step.a, step.o, step.r
end
