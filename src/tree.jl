struct SparseParticleTree{S,A,O}
    Nh::Vector{Int}
    Nha::Vector{Int}# Number of times a history-action node has been visited
    Qha::Vector{Float64} # Map ba node to associated Q value

    b::Vector{PFTBelief{S}}
    b_children::Vector{Vector{Tuple{A,Int}}} # b_idx => [(a,ba_idx), ...]
    b_rewards::Vector{Float64}# Map b' node index to immediate reward associated with trajectory bao where b' = τ(bao)

    bao_children::Dict{Tuple{Int,O},Int} # (ba_idx,O) => bp_idx
    ba_children::Vector{Vector{Int}} # ba_idx => [bp_idx, bp_idx, bp_idx, ...]

    function SparseParticleTree{S,A,O}(sz::Int; check_repeat_obs::Bool=true) where {S,A,O}
        return new(
            sizehint!(Int[],sz),
            sizehint!(Int[],sz),
            sizehint!(Float64[], sz),

            sizehint!(PFTBelief{S}[],sz),
            sizehint!(Vector{Tuple{A,Int}}[], sz),
            sizehint!(Float64[], sz),

            sizehint!(Dict{Tuple{Int,O},Int}(), check_repeat_obs ? sz : 0),
            sizehint!(Vector{Int}[], sz)
            )
    end
end


function insert_belief!(planner::SparsePFTPlanner, b::PFTBelief, ba_idx, o, r)
    tree = planner.tree
    n_b = length(tree.b)+1
    push!(tree.b, b)
    push!(tree.Nh, 0)
    push!(tree.b_rewards, r)
    push!(tree.ba_children[ba_idx],n_b)

    push!(tree.b_children, Int[])

    if planner.sol.check_repeat_obs
        tree.bao_children[(ba_idx,obs)] = n_b
    end
    return n_b
end

function initial_belief(rng::AbstractRNG, pomdp::POMDP{S}, b, n_p::Int) where S
    s = Vector{S}(undef, n_p)
    w_i = inv(n_p)
    w = fill(w_i, n_p)
    term_ws = 0.0

    for i in eachindex(s)
        s_i = rand(rng,b)
        s[i] = s_i
        !isterminal(pomdp, s_i) && (term_ws += w_i)
    end

    return PFTBelief(s, w, term_ws)
end

function insert_root!(planner::SparsePFTPlanner, b)
    particle_b = initial_belief(planner.sol.rng, planner.pomdp, b, planner.sol.n_particles)
    push!(planner.tree.b, particle_b)
    push!(planner.tree.b_children, Int[])
    push!(planner.tree.Nh, 0)
    push!(planner.tree.b_rewards, 0.0)
end

function insert_action!(planner::SparsePFTPlanner, tree::SparseParticleTree, b_idx, a)
    n_ba = length(tree.ba_children)+1
    push!(tree.b_children[b_idx], (a,n_ba))
    push!(tree.ba_children, Int[])
    push!(tree.Nha, 0)
    push!(tree.Qha, 0.0)
end

function Base.empty!(tree::SparseParticleTree)
    empty!(tree.Nh)
    empty!(tree.Nha)
    empty!(tree.Qha)

    empty!(tree.b)
    empty!(tree.b_children)
    empty!(tree.b_rewards)

    empty!(tree.bao_children)
    empty!(tree.ba_children)
end
