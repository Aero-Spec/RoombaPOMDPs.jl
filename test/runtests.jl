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
        # LidarPOMDP (continuous) error branches:
        m_lidar = RoombaPOMDP(sensor=Lidar(), mdp=RoombaMDP())
        @test_throws ErrorException RoombaPOMDPs.n_observations(m_lidar)   # module-qualified!
        @test_throws ErrorException POMDPs.observations(m_lidar)

        # DiscreteLidarPOMDP happy path:
        disc_points = [0.3, 0.6, 1.0]  # -> 4 bins
        s_disc = DiscreteLidar(Lidar().ray_stdev, disc_points, zeros(Float64, length(disc_points)+1))
        m_dlidar = RoombaPOMDP(sensor=s_disc, mdp=RoombaMDP())

        sp = rand(rng, initialstate(m_dlidar))

        d = POMDPs.observation(m_dlidar, sp)  # SparseCat over 1:4
        @test n_observations(m_dlidar) == length(disc_points) + 1
        @test collect(POMDPs.observations(m_dlidar)) == collect(1:n_observations(m_dlidar))
        @test all(o -> POMDPs.obsindex(m_dlidar, o) == o, support(d))
        @test isapprox(sum(pdf.(Ref(d), support(d))), 1.0; atol=1e-9)
        @test all(p -> p ≥ 0.0, pdf.(Ref(d), support(d)))

        # observation over Int state overload
        si = POMDPs.convert_s(Int, sp, m_dlidar)
        d2 = POMDPs.observation(m_dlidar, si)
        @test length(support(d2)) == n_observations(m_dlidar)
        @test isapprox(sum(pdf.(Ref(d2), support(d2))), 1.0; atol=1e-9)
    end

    @testset "Particle filter update & resampling" begin
        m = RoombaPOMDP(sensor=Lidar(), mdp=RoombaMDP())

        n = 200
        b = ParticleCollection([rand(rng, initialstate(m)) for _ in 1:n])

        up = RoombaParticleFilter(m, n, 0.05, 0.05, nothing, rng)

        s = rand(rng, initialstate(m))
        a = RoombaAct(0.3, 0.0)
        sp = rand(transition(m, s, a))
        o = rand(rng, observation(m, sp))  # Float64

        bnew = POMDPs.update(up, b, a, o)
        @test bnew isa ParticleCollection
        @test length(collect(particles(bnew))) == n
        @test any(!isterminal(m, p) for p in particles(bnew))
    end
end
