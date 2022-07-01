function POMDPModelTools.action_info(planner::SparsePFTPlanner, b)
    t0 = time()

    (;sol,pomdp) = planner
    (;max_iter, max_time, max_depth) = sol

    A = actiontype(pomdp)

    empty!(planner.tree)
    insert_root!(planner, b)

    iter = 0
    while (time()-t0 < max_time) && (iter < max_iter)
        search(planner, 1, max_depth)
        iter += 1
    end

    a, a_idx = ucb_action(planner, 1)

    return a, (
        n_iter = iter,
        tree = planner.tree,
        time = time() - t0
        )
end

function POMDPs.action(planner::PFTDPWPlanner, b)
    return first(action_info(planner, b))
end
