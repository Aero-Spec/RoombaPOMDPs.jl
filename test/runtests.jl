using Test
using Random
using POMDPs
using POMDPTools
using Distributions
using ParticleFilters: ParticleCollection, particles
using RoombaPOMDPs

# Simple always-forward policy (kept around; not used with stepthrough)
struct ForwardPolicy end
POMDPs.action(::ForwardPolicy, ::Any) = RoombaAct(0.2, 0.0)

@testset "RoombaPOMDPs.jl" begin
    rng = MersenneTwister(1)

    @testset "Constructors and basics" begin
        mdp = RoombaMDP()
        @test mdp isa RoombaMDP

        pomdp_bumper = RoombaPOMDP(sensor=Bumper(), mdp=mdp)
        @test pomdp_bumper isa BumperPOMDP

        pomdp_lidar = RoombaPOMDP(sensor=Lidar(), mdp=mdp)
        @test pomdp_lidar isa LidarPOMDP

        s0 = rand(rng, initialstate(pomdp_lidar))
        @test s0 isa RoombaState
        @test isfinite(s0.x) && isfinite(s0.y) && isfinite(s0.theta)
    end

    @testset "Transition and reward" begin
        mdp = RoombaMDP()
        s = RoombaState(0.0, 0.0, 0.0, 0.0)
        a = RoombaAct(0.5, 0.2)

        sp = rand(transition(mdp, s, a))
        @test sp isa RoombaState

        r = reward(mdp, s, a, sp)
        @test r ≤ mdp.time_pen + max(mdp.goal_reward, 0.0)
    end

   @testset "Observation API coverage" begin
    rng = MersenneTwister(0)

    # --- LidarPOMDP (continuous) error branches ---
    m_lidar = RoombaPOMDP(sensor=Lidar(), mdp=RoombaMDP())
    @test_throws ErrorException RoombaPOMDPs.n_observations(m_lidar) # module-qualified
    @test_throws ErrorException POMDPs.observations(m_lidar)

    # --- DiscreteLidarPOMDP with CONTINUOUS state space (RoombaState path) ---
    disc_points = [0.3, 0.6, 1.0]              # -> 4 bins total
    s_disc = DiscreteLidar(Lidar().ray_stdev,   # use 3-arg ctor; pass buffer
                           disc_points,
                           zeros(Float64, length(disc_points)+1))
    m_dlidar = RoombaPOMDP(sensor=s_disc, mdp=RoombaMDP())

    sp = rand(rng, initialstate(m_dlidar))     # RoombaState
    d = POMDPs.observation(m_dlidar, sp)       # SparseCat over 1:4

    nobs = RoombaPOMDPs.n_observations(m_dlidar)  # module-qualified
    @test nobs == length(disc_points) + 1
    @test collect(POMDPs.observations(m_dlidar)) == collect(1:nobs)
    @test all(o -> POMDPs.obsindex(m_dlidar, o) == o, support(d))
    @test isapprox(sum(pdf.(Ref(d), support(d))), 1.0; atol=1e-9)
    @test all(p -> p ≥ 0.0, pdf.(Ref(d), support(d)))

    # --- DiscreteLidarPOMDP with DISCRETE state space (Int path) ---
    # Choose sizes that satisfy x_step==y_step: (num_x_pts-1) = 8/5*(num_y_pts-1).
    # Example: num_y_pts=6 -> (6-1)=5, so num_x_pts=9.
    ss_disc = DiscreteRoombaStateSpace(9, 6, 5)
    mdp_disc = RoombaMDP(sspace=ss_disc)                 # room will be consistent with sspace
    m_dlidar_disc = RoombaPOMDP(sensor=s_disc, mdp=mdp_disc)

    si = rand(rng, POMDPs.states(m_dlidar_disc))         # Int state index
    d2 = POMDPs.observation(m_dlidar_disc, si)
    nobs2 = RoombaPOMDPs.n_observations(m_dlidar_disc)
    @test length(support(d2)) == nobs2
    @test isapprox(sum(pdf.(Ref(d2), support(d2))), 1.0; atol=1e-9)
end
