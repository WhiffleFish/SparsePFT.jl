using SparsePFT
using POMDPs
using POMDPModels
using Test

@testset "smoke tests" begin
    pomdp = TigerPOMDP()
    sol = SparsePFTSolver()
    policy = solve(sol, pomdp)
    b0 = initialstate(pomdp)
    a = action(policy, b0)
    @test a ∈ actions(pomdp)
end
