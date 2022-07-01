Base.@kwdef struct RandomRollout{RNG<:AbstractRNG}
    rng::RNG = Random.default_rng()
end

struct RandomRolloutEstimator{A, RNG <: AbstractRNG}
    actions::A
    rng::RNG
end

function MCTS.convert_estimator(r::RandomRollout, ::Any, p::POMDP)
    return RandomRolloutEstimator(actions(p), r.rng)
end

function MCTS.estimate_value(est::RandomRolloutEstimator, pomdp::POMDP, b::PFTBelief, d::Int)
    v = 0.0
    for (s,w) in weighted_particles(b)
        v += w*rollout(est, pomdp, s, d)
    end
    return v
end

function rollout(est::RandomRolloutEstimator, pomdp::POMDP, s, d::Int)
    rng = est.rng
    γ = discount(pomdp)

    disc = 1.0
    r_total = 0.0
    step = 1

    while !isterminal(pomdp, s) && step ≤ d

        a = rand(est.actions)

        sp, r = @gen(:sp,:r)(pomdp, s, a, rng)

        r_total += disc*r

        s = sp

        disc *= γ
        step += 1
    end

    return r_total
end
