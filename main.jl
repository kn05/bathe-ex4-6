using SparseArrays

using Symbolics
using SymbolicNumericIntegration

# 1. element 별로 K matrix 구하는 부분 함수로 만들기
# 2-1. K_global로 합치기
# 2-2. impose boundary condition
# 3. KU = R 이용하여 U 구하고 U plot하기

LM_2 = [11, 5, 3, 9, 12, 6, 4, 10]

# calculate local stiffness matrix
function calc_K_m(LM::Vector{Int64})
    # Will be modified to receive it as a function argument later
    E = 1.
    nu = 0.5
    x_min, x_max = -1., 1.
    y_min, y_max = -1., 1.

    @variables x y

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
    integrand = B_m_dense' * C_m * B_m_dense
    K_m_sym = integrate(integrate(integrand, x; symbolic=true, detailed=false), y; symbolic=true, detailed=false)
    K_m_upper = substitute(K_m_sym, Dict([x => x_max, y => y_max]))
    K_m_lower = substitute(K_m_sym, Dict([x => x_min, y => y_min]))
    K_m = K_m_upper - K_m_lower
    
    return K_m
end


