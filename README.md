# Bathe-ex4-6
This project is for practicing the Finite Element Method with Example 4.6 from "Finite Element Procedures" by Bathe.

## Example 4.6 
Consider the analysis of the cantilever plate shown in Fig. E4.6. To illustrate the analysis technique, use the coarse finite element idealization given in the figure (in a practical analysis more finite elements must be employed (see Section 4.3). Establish the matrices H(2), B(2), and C(2).

The cantilever plate is acting in plane stress conditions. For an isotropic linear elastic material the stress-strain matrix is defined using Young’s modulus E and Poisson’s ratio v (see Table 4.3),
$$ 
\mathrm{C}^{(2)} = \frac{E}{1-\nu^{2}}  
\begin{bmatrix} 
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1-\nu}{2}
\end{bmatrix}
$$

