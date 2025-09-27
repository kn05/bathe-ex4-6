using SparseArrays
using LinearAlgebra
using Plots
using Symbolics
using SymbolicNumericIntegration

LM_1 = [9, 3, 1, 7, 10, 4, 2, 8]
LM_2 = [11, 5, 3, 9, 12, 6, 4, 10]
LM_3 = [15, 9, 7, 13, 16, 10, 8, 14]
LM_4 = [17, 11, 9, 15, 18, 12, 10, 16]

# calculate local stiffness matrix
function calc_K_m(LM::Vector{Int64})
    # Will be modified to receive it as a function argument later
    # E = 2.07e13  # P, 
    # nu = 0.30
    x_min, x_max = -1., 1.
    y_min, y_max = -1., 1.

    @variables x y E nu

    C_m = E / (1 - nu^2) * [
        1 nu 0;
        nu 1 0
        0 0 (1-nu)/2
    ]

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


    # K_m = int_V B^T C B dV
    # !! This integrate have error, have to fix it
    integrand = B_m_dense' * C_m * B_m_dense
    integral_x = integrate(integrand, x; symbolic=true, detailed=false)

    eval_x = substitute(integral_x, Dict([x => x_max])) - substitute(integral_x, Dict([x => x_min]))
    simple_eval_x = simplify(eval_x)
    
    integral_y = integrate(simple_eval_x, y; symbolic=true, detailed=false)
    simple_integral_y = simplify(integral_y)
    
    println(simple_integral_y)

    eval_y = substitute(simple_integral_y, Dict([y => y_max])) - substitute(simple_integral_y, Dict([y => y_min]))
    K_m = substitute(eval_y, Dict([E => 1., nu => 0.5]))
    
    return K_m
end

K_global = spzeros(18, 18)
for LM in [LM_1, LM_2, LM_3, LM_4]
    K_m = calc_K_m(LM)
    K_global += K_m
end

R = zeros(18)
R[18] = -1.

K_constrained = copy(K_global)
R_constrained = copy(R)

# Apply boundary conditions using the penalty method
fixed_dofs = [1, 2, 3, 4, 5, 6]
penalty = 1.0e24 # large number
for i in fixed_dofs
    K_constrained[i, i] += penalty
end

U = inv(K_constrained) * R_constrained

U_normal = U / maximum(abs.(U))

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
    ylims=(-1.5, 5.5),
    label="Original Position",
    title="Node Displacements",
    legend=false
)

scatter!(plt, displaced_x_coords, displaced_y_coords,
    label="Displaced Position"
)

display(plt)
