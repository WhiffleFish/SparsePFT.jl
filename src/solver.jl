Base.@kwdef struct SparsePFTSolver{RNG<:AbstractRNG} <: Solver
    tree_queries::Int      = 1_000
    max_time::Float64      = Inf # (seconds)
    max_depth::Int         = 20
    n_particles::Int       = 100
    c::Float64             = 1.0
    k_o::Float64           = 10.0
    k_a::Float64           = 5.0
    rng::RNG               = Random.default_rng()
    value_estimator::Any   = RandomRollout()
    check_repeat_obs::Bool = true
    enable_action_pw::Bool = false
end

struct SparsePFTPlanner{SOL<:SparsePFTSolver, VE, P<:POMDP, TREE} <: Policy
    sol::SOL
    tree::TREE
    pomdp::P
    value_estimator::VE
end

function POMDPs.solve(sol::SparsePFTSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    solved_ve = MCTS.convert_estimator(sol.value_estimator, sol, pomdp)
    sz = min(sol.tree_queries, 100_000)
    return SparsePFTPlanner(
        sol,
        SparseParticleTree{S,A,O}(sz, check_repeat_obs=sol.check_repeat_obs),
        pomdp,
        solved_ve
    )
end
