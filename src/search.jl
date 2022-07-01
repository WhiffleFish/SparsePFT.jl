function search(planner::PFTDPWPlanner, b_idx::Int, d::Int)
    (;tree, pomdp, sol) = planner
    γ = discount(pomdp)

    if iszero(d) || tree.b[b_idx].non_terminal_ws < eps()
        return 0.0
    end

    a, ba_idx = ucb_action(planner, b_idx)
    if length(tree.ba_children[ba_idx]) ≤ sol.k_o
        sample_s = non_terminal_sample(sol.rng, pomdp, tree.b[b_idx])
        o = @gen(:o)(pomdp, sample_s, a, planner.sol.rng)

        if !haskey(tree.bao_children, (ba_idx, o))
            bp, _, r = τ(planner, b, a, o)
            insert_belief!(tree, bp, ba_idx, o, r, planner)
            ro = MCTS.estimate_value(planner.solved_VE, pomdp, bp, d-1)
            total = r + γ*ro
        else
            bp_idx = tree.bao_children[(ba_idx,o)]
            push!(tree.ba_children[ba_idx], bp_idx)
            r = tree.b_rewards[bp_idx]
            total = r + γ*search(planner, bp_idx, d-1)
        end
    else
        bp_idx = rand(sol.rng, tree.ba_children[ba_idx])
        r = tree.b_rewards[bp_idx]
        total = r + γ*search(planner, bp_idx, d-1)
    end

    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1
    tree.Qha[ba_idx] = tree.Qha[ba_idx] + (total - tree.Qha[ba_idx]) / tree.Nha[ba_idx]

    return total
end
