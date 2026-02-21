using SparseArrays
using LinearAlgebra
using Plots
using Integrals
using Cubature

LM_1 = [9, 3, 1, 7, 10, 4, 2, 8]
LM_2 = [11, 5, 3, 9, 12, 6, 4, 10]
LM_3 = [15, 9, 7, 13, 16, 10, 8, 14]
LM_4 = [17, 11, 9, 15, 18, 12, 10, 16]

function calc_B_mat(x::Float64, y::Float64, LM::Vector{Int64})
    B = 1 / 4 * [
        (1+y) -(1 + y) -(1 - y) (1-y) 0 0 0 0;
        0 0 0 0 (1+x) (1-x) -(1 - x) -(1 + x);
        (1+x) (1-x) -(1 - x) -(1 + x) (1+y) -(1 + y) -(1 - y) (1-y)
    ]

    B_m_sparse = begin
        rows = repeat([1, 2, 3], inner=length(LM))
        cols = [LM; LM; LM]
        vals = [B[1, :]; B[2, :]; B[3, :]]
        sparse(rows, cols, vals, 3, 18)
    end
    B_m_dense = Matrix(B_m_sparse)

    return B_m_dense
end

# calculate local stiffness matrix
function calc_K_m(LM::Vector{Int64})
    # Will be modified to receive it as a function argument later
    # E = 2.07e13  # P, 
    # nu = 0.30
    x_min, x_max = -1., 1.
    y_min, y_max = -1., 1.

    E = 2.07e12
    nu = 0.30

    C_m = E / (1 - nu^2) * [
        1 nu 0;
        nu 1 0
        0 0 (1-nu)/2
    ]

    # K_m = int_V B^T C B dV
    BtCB(u, p) = calc_B_mat(u[1], u[2], LM)' * C_m * calc_B_mat(u[1], u[2], LM)
    domain = ([-1., -1.], [1., 1.])
    prob = IntegralProblem(BtCB, domain)
    K_m = solve(prob, CubatureJLh(); reltol=1e-3, abstol=1e-3)

    return K_m
end

function impose_constrain(method::Val{:penalty}, K, fixed_dofs, penalty=1.0e24)
    K_constrained = copy(K)
    for i in fixed_dofs
        K_constrained[i, i] += penalty
    end
end

function impose_constrain(method::Val{:exact}, K, fixed_dofs, R)
    K_constrained = copy(K)

    # U_a : unknown displacement, U_b: prescribed displacement
    # at here, fixed dof means zero displacement
    b = fixed_dofs
    a = setdiff(collect(1:size), b)
    U_b = zeros(length(fixed_dofs))
    K_aa = @view K[a, a]
    K_ab = @view K[a, b]
    K_ba = @view K[b, a]
    K_bb = @view K[b, b]
    U_a = inv(K_aa) * (R_aa - K_ab * U_b)
    R_r = K_ba*U_a + K_bb*U_b - R

    return U_a, R_r
end

function main()
    K_global = spzeros(18, 18)
    for LM in [LM_1, LM_2, LM_3, LM_4]
        K_m = calc_K_m(LM)
        K_global += K_m
    end

    R = zeros(18)
    R[18] = -1.

    K_constrained = copy(K_global)
    fixed_dofs = [1, 2, 3, 4, 5, 6]

    K_constrained = impose_constrain(K, fixed_dofs, Val(:panalty))

    U = inv(K_constrained) * R_constrained

    U_normal = U / maximum(abs.(U)) * 0.5

    x = [
        (0., 0.), (0., 2.), (0., 4.),
        (2., 0.), (2., 2.), (2., 4.),
        (4., 0.), (4., 2.), (4., 4.)
    ]

    dx = [(U_normal[2i-1], U_normal[2i]) for i in 1:9]
    displaced_x = map((p, dp) -> (p[1] + dp[1], p[2] + dp[2]), x, dx)

    x_coords = [p[1] for p in x]
    y_coords = [p[2] for p in x]
    displaced_x_coords = [p[1] for p in displaced_x]
    displaced_y_coords = [p[2] for p in displaced_x]


    plt = scatter(x_coords, y_coords,
        aspect_ratio=1,
        xlims=(-0.5, 4.5),
        ylims=(-0.5, 4.5),
        label="Original Position",
        title="Node Displacements",
        legend=false
    )

    scatter!(plt, displaced_x_coords, displaced_y_coords,
        label="Displaced Position"
    )

    display(plt)

    return plt
end

main()

