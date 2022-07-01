function ucb_action(planner, b_idx)
    tree = planner.tree
    lnh = log(tree.Nh[b_idx])
    local opt_a::A
    max_ucb = -Inf
    opt_idx = 0

    for (a,ba_idx) in tree.b_children[b_idx]
        Nha = tree.Nha[ba_idx]
        iszero(Nha) && return a, ba_idx
        ucb = tree.Qha[ba_idx] + c*sqrt(lnh / Nha)

        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a, opt_idx
end
