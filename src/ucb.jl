function _ucb_action(planner, b_idx)
    (;tree, sol) = planner
    A = actiontype(planner.pomdp)
    lnh = log(tree.Nh[b_idx])
    local opt_a::A
    max_ucb = -Inf
    opt_idx = 0

    for (a,ba_idx) in tree.b_children[b_idx]
        Nha = tree.Nha[ba_idx]
        iszero(Nha) && return a, ba_idx
        ucb = tree.Qha[ba_idx] + sol.c*sqrt(lnh / Nha)

        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a, opt_idx
end

function ucb_action(planner, b_idx)
    (;tree, sol) = planner
    A = actions(planner.pomdp)
    a = first(A)
    ba_idx = 0

    if planner.sol.enable_action_pw
        if length(tree.b_children[b_idx]) < sol.k_a
            a, ba_idx = rand(sol.rng, A), length(tree.ba_children)+1
            insert_action!(tree, b_idx, a)
        else
            a, ba_idx = _ucb_action(planner, b_idx)
        end
    else
        if isempty(tree.b_children[b_idx])
            n_ba = length(tree.ba_children)
            a_idx = rand(sol.rng, 1:length(A))
            for _a in A
                insert_action!(tree, b_idx, _a)
            end
            a, ba_idx = A[a_idx], n_ba+a_idx
        else
            a, ba_idx = _ucb_action(planner, b_idx)
        end
    end
    return a, ba_idx
end
