module RoombaPOMDPs

using POMDPs
import POMDPs: initialize_belief   # Only if you extend it

using Distributions
using StaticArrays
using Parameters
using POMDPTools
using Statistics
using Graphics
using Cairo
using Random
import ParticleFilters
import POMDPTools: render
using ParticleFilters: resample

# Export all main API symbols for users of your package
export
    RoombaState,
    RoombaAct,
    RoombaMDP,
    RoombaPOMDP,
    RoombaModel,
    Bumper,
    BumperPOMDP,
    Lidar,
    LidarPOMDP,
    DiscreteLidar,
    DiscreteLidarPOMDP,
    RoombaParticleFilter,
    get_goal_xy,
    wrap_to_pi,
    ContinuousRoombaStateSpace,
    DiscreteRoombaStateSpace,
    render

# Include the submodules/files in the right order (so dependencies are loaded before used)
include("line_segment_utils.jl")
include("env_room.jl")
include("roomba_env.jl")
include("filtering.jl")

end # module RoombaPOMDPs
