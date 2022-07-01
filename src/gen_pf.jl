function gen_pf(planner, b::PFTBelief{S}, a, o) where S
    rng = planner.sol.rng
    pomdp = planner.pomdp
    N = n_particles(b)
    ρ = 0.0

    s′ = Vector{S}(undef, N)
    w′ = Vector{Float64}(undef, N)

    for (i,(s,w)) in enumerate(weighted_particles(b))
        if !isterminal(pomdp, s)
            (sp, r) = @gen(:sp,:r)(pomdp, s, a, rng)
        else
            (sp,r) = (s, 0.0)
        end

        w = weight(b, i)
        s′[i] = sp
        w′[i] = w*pdf(POMDPs.observation(pomdp, s, a, sp), o)

        ρ += r*w
    end
    return PFTBelief(s′, w′, pomdp), ρ
end
