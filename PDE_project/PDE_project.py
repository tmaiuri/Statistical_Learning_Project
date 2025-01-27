import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

########################################### 1/ Geometry ##############################################################

def generate_rectangle_mesh(Lx: float, Ly: float, nb_sub_h: int, nb_sub_v: int):
    """
    Generates a uniform triangular mesh for a rectangular domain

    Args:
        Lx (float): Horizontal length of the domain
        Ly (float): Vertical length
        nb_sub_h (int): Number of horizontal subdivisions
        nb_sub_v (int): Number of vertical subdivisions

    Returns:
        tuple: (vtx, elt)
            vtx (ndarray): Coordinate matrix
            elt (ndarray): Connectivity matrix
    """
    nb_vtx = (nb_sub_h + 1) * (nb_sub_v + 1)
    nb_elt = 2 * nb_sub_h * nb_sub_v
    pas_h = Lx / nb_sub_h
    pas_v = Ly / nb_sub_v

    vtx = np.zeros((nb_vtx, 2))
    for i in range(nb_sub_v + 1):
        for j in range(nb_sub_h + 1):
            idx = j + i * (nb_sub_h + 1)
            vtx[idx, :] = [j * pas_h, i * pas_v]

    elt = np.zeros((nb_elt, 3), dtype=int)
    for i in range(nb_sub_v):
        for j in range(nb_sub_h):
            k1 = j + i * (nb_sub_h + 1)
            k2 = k1 + nb_sub_h + 2
            elt[2 * j + 2 * nb_sub_h * i, :] = [k1, k1 + 1, k2]
            elt[2 * j + 2 * nb_sub_h * i + 1, :] = [k1, k2, k2 - 1]

    return vtx, elt

def geometric_refinement(vtx: np.ndarray, elt: np.ndarray, r: int):
    """
    Locally refines the mesh around the origin

    Args:
        vtx (ndarray): Coordinate matrix
        elt (ndarray): Connectivity matrix
        r (int): Number of refinement levels

    Returns:
        tuple: (ref_vtx, ref_elt)
            ref_vtx (ndarray): Vertices of the refined mesh
            ref_elt (ndarray):  Connectivity of refined elements
    """
    if (r <= 0):
        return vtx, elt

    for k in range(1, r+1):
        refined_elt = []
        refined_vtx = []
        n = np.size(vtx, 0)

        for it_vtx in vtx:
            x = it_vtx[0]
            y = it_vtx[1]
            refined_vtx.append([x, y])

        def get_midpoints(s, t):
            x1 = vtx[s][0]
            y1 = vtx[s][1]

            x2 = vtx[t][0]
            y2 = vtx[t][1]

            return (x1+x2)/2, (y1+y2)/2
            
        for a, b, c in elt:
            if (a == 0):
                new_x1, new_y1 = get_midpoints(a, b)
                new_x2, new_y2 = get_midpoints(a, c)
 
                if (c > b):
                    refined_elt += [[0, n, n+2], [n, b, n+2], [b, c, n+2]]
                    refined_vtx.append([new_x1, new_y1])
                else:
                    refined_elt += [[0, n+2, n+1], [n+2, b, c], [n+1, n+2, c]]
                    refined_vtx += [[new_x2, new_y2], [new_x1, new_y1]]
            else :
                refined_elt.append([a, b, c])

        ref_vtx, ref_elt = np.asarray(refined_vtx), np.asarray(refined_elt)
        return geometric_refinement(ref_vtx, ref_elt, r-k)

    return vtx, elt

def plot_mesh(vtx: np.ndarray, elt: np.ndarray, val: np.ndarray = None):
    """
    Displays a triangular mesh

    Args:  
        vtx (ndarray): Vertex coordinates
        elt (ndarray): Element connectivity
        val (ndarray, optional): Values associated with vertices for colouring
    """
    if val is not None:
        plt.tricontourf(vtx[:, 0], vtx[:, 1], elt, val, levels=20)
        plt.colorbar()
    plt.triplot(vtx[:, 0], vtx[:, 1], elt, color='black', linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")

########################################### 2/ Problem ###############################################################

""" 
Assembly of elementary matrices associated with each term appearing in 
the sesquilinear form
"""

def compute_volume(vtx: np.ndarray) -> float:
    """
    Calculates the volume of a triangle given by its vertices

    Args:
        vtx (ndarray): Coordinates of the vertices of the triangle (3x2)

    Returns:
        float: Area of the triangle
    """
    if vtx.shape == (3, 2):
        return 0.5 * abs(np.linalg.det(np.array([
            [vtx[1, 0] - vtx[0, 0], vtx[2, 0] - vtx[0, 0]],
            [vtx[1, 1] - vtx[0, 1], vtx[2, 1] - vtx[0, 1]]
        ])))
    raise ValueError("vtx must be 3x2 dimensional")

def local_mass_matrix(vtx: np.ndarray) -> np.ndarray:
    """
    Calculates the local mass matrix for a triangular element

    Args:
        vtx (ndarray): Coordinates of the vertices of the triangle (3x2)

    Returns:
        ndarray: Local mass matrix (3x3)
    """
    volume = compute_volume(vtx)
    return (volume / 12.0) * (np.ones((3, 3)) + np.eye(3))

def assemble_mass_matrix(vtx: np.ndarray, elt: np.ndarray) -> coo_matrix:
    """
    Assembles the global mass matrix

    Args:
        vtx (ndarray): Vertex coordinates
        elt (ndarray): Connectivity matrix

    Returns:
        coo_matrix: Global mass matrix
    """
    n_vtx = vtx.shape[0]
    row, col, data = [], [], []

    for tri in elt:
        triangle_coords = vtx[tri]
        m_local = local_mass_matrix(triangle_coords)

        for i in range(3):
            for j in range(3):
                row.append(tri[i])
                col.append(tri[j])
                data.append(m_local[i, j])

    return coo_matrix((data, (row, col)), shape=(n_vtx, n_vtx))

def local_stiffness_matrix(vtx: np.ndarray, elt: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Calculates the local stiffness matrix for a triangular element

    Args:
        vtx (ndarray): Coordinates of the vertices of the triangle (3x2)
        mu (np.ndarray): Mu coefficients at the vertices

    Returns:
        ndarray: Local stiffness matrix (3x3)
    """

    d = elt.shape[0]
    v0, v1, v2 = vtx[elt[0]], vtx[elt[1]], vtx[elt[2]]
    n = np.array([
        [v1[1] - v2[1], v2[0] - v1[0]],  # Normal to edge 2-3
        [v2[1] - v0[1], v0[0] - v2[0]],  # Normal to edge 3-1
        [v0[1] - v1[1], v1[0] - v0[0]]   # Normal to edge 1-2
    ])

    g = np.array([n[i] / np.dot(vtx[elt[i]] - vtx[elt[(i + 1) % d]], n[i]) for i in range(d)])
    return compute_volume(vtx[elt]) * np.dot(g, g.T) * (mu[elt[0]] + mu[elt[1]] + mu[elt[2]]) / 3

def assemble_stiffness_matrix(vtx: np.ndarray, elt: np.ndarray, mu: np.ndarray) -> coo_matrix:
    """
    Assembles the global stiffness matrix

    Args:
        vtx (ndarray): Vertex coordinates
        elt (ndarray): Connectivity matrix
        mu (ndarray): Mu coefficients at vertices

    Returns:
        coo_matrix: Global stiffness matrix
    """
    n_vtx = vtx.shape[0]
    row, col, data = [], [], []

    for tri in elt:
        k_local = local_stiffness_matrix(vtx, tri, mu)

        for i in range(3):
            for j in range(3):
                row.append(tri[i])
                col.append(tri[j])
                data.append(k_local[i, j])

    return coo_matrix((data, (row, col)), shape=(n_vtx, n_vtx))

def global_matrix(vtx: np.ndarray, elt: np.ndarray, mu: np.ndarray) -> coo_matrix:
    """
    Assembles the global matrix of the variational problem (mass + stiffness).

    Args:
        vtx (ndarray): Coordinates matrix
        elt (ndarray): Connectivity matrix
        mu (ndarray): Mu coefficients at vertices.

    Returns:
        coo_matrix: Global matrix of the system.
    """
    m = assemble_mass_matrix(vtx, elt)
    k = assemble_stiffness_matrix(vtx, elt, mu)
    return m + k

########################################### 3/ Second member ##########################################################

"""
Adding functions, their derivatives and assembling the second member
"""

def check_conditions(x, y, Lx, Ly, alpha=None):
    if alpha is not None and alpha <= 0.5:
        raise ValueError("alpha must be > 0.5")
    if x < 0 or x > Lx or y < 0 or y > Ly:
        raise ValueError("(x, y) must be in Omega")

def u_ex(x: float, y: float, Lx: float, Ly: float, alpha: float) -> float:
    return (x * y) ** alpha * (x - Lx) * (y - Ly)

def mu(x: float, y: float, Lx: float, Ly: float) -> float:
    return 2 + np.sin(2 * np.pi * x / Lx) * np.sin(4 * np.pi * y / Ly)

def d_x_mu(x: float, y: float, Lx: float, Ly: float) -> float:
    return (2 * np.pi / Lx) * np.cos(2 * np.pi * x / Lx) * np.sin(4 * np.pi * y / Ly)

def d_y_mu(x: float, y: float, Lx: float, Ly: float) -> float:
    return (4 * np.pi / Ly) * np.sin(2 * np.pi * x / Lx) * np.cos(4 * np.pi * y / Ly)

def d_x_u(x: float, y: float, Lx: float, Ly: float, alpha: float) -> float:
    if alpha == 1:
        return (2 * x - Lx) * y * (y - Ly)
    return (x ** (alpha - 1)) * (y ** alpha) * ((alpha + 1) * x - alpha * Lx) * (y - Ly)

def d_y_u(x: float, y: float, Lx: float, Ly: float, alpha: float) -> float:
    if alpha == 1:
        return (2 * y - Ly) * x * (x - Lx)
    return (y ** (alpha - 1)) * (x ** alpha) * ((alpha + 1) * y - alpha * Ly) * (x - Lx)

def d2_x_u(x: float, y: float, Lx: float, Ly: float, alpha: float) -> float:
    if alpha == 1:
        return 2 * y * (y - Ly)
    return (x ** (alpha - 2)) * (y ** alpha) * ((alpha + 1) * alpha * x - (alpha - 1) * alpha * Lx) * (y - Ly)

def d2_y_u(x: float, y: float, Lx: float, Ly: float, alpha: float) -> float:
    if alpha == 1:
        return 2 * x * (x - Lx)
    return (y ** (alpha - 2)) * (x ** alpha) * ((alpha + 1) * alpha * y - (alpha - 1) * alpha * Ly) * (x - Lx)

def f(x: float, y: float, Lx: float, Ly: float, alpha: float) -> float:
    check_conditions(x, y, Lx, Ly, alpha)
    mu_val = mu(x, y, Lx, Ly)
    d2_x_u_val = d2_x_u(x, y, Lx, Ly, alpha)
    d2_y_u_val = d2_y_u(x, y, Lx, Ly, alpha)
    d_x_mu_val = d_x_mu(x, y, Lx, Ly)
    d_y_mu_val = d_y_mu(x, y, Lx, Ly)
    d_x_u_val = d_x_u(x, y, Lx, Ly, alpha)
    d_y_u_val = d_y_u(x, y, Lx, Ly, alpha)
    u_val = u_ex(x, y, Lx, Ly, alpha)

    return - mu_val * (d2_x_u_val + d2_y_u_val) - d_x_mu_val * d_x_u_val - d_y_mu_val * d_y_u_val + u_val

def Second_membre(vtx: np.ndarray, elt: np.ndarray, Lx: float, Ly: float, alpha: float) -> np.ndarray:
    n = vtx.shape[0]
    m = assemble_mass_matrix(vtx, elt)
    f_test = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = vtx[i][0]
        y = vtx[i][1]
        if (x == 0.) or (y == 0.) or (x == Lx) or (y == Ly):
             f_test[i] = 0.
        else : 
            f_test[i] = f(x, y, Lx, Ly, alpha)
    return m.dot(f_test)

########################################### 4/ Resolution ##############################################################

def generate_sparse_diagonal_matrix(vtx: np.ndarray, Lx: float, Ly: float) -> coo_matrix:
    diagonal_values = np.array([0 if (v[0] == 0) 
                                    or (v[0] == Lx)
                                    or (v[1] == 0)
                                    or (v[1] == Ly)
                                    else 1 for v in vtx])
    d = diags(diagonal_values, offsets=0, format='coo')
    
    return d

def solve(Lx: float, Ly: float, nb_sub_h: int, nb_sub_v: int, alpha: float) -> np.ndarray:
    """
    Solves the global linear system Ax = b

    Args:
        vtx (ndarray): Coordinates matrix
        elt (ndarray): Connectivity matrix
        mu (ndarray): Mu coefficients at vertices
        f (ndarray): Global second member

    Returns:
        ndarray: Approximate solution
    """
    vtx, elt = generate_rectangle_mesh(Lx, Ly, nb_sub_h, nb_sub_v)
    n = vtx.shape[0]
    mu_test = np.zeros(n)
    d = generate_sparse_diagonal_matrix(vtx, Lx, Ly)
    I = sp.sparse.eye(n)

    for i in range(n):
        x = vtx[i][0]
        y = vtx[i][1]
        mu_test[i] = mu(x, y, Lx, Ly)
    a = global_matrix(vtx, elt, mu_test)
    a = d@a@d + I - d
    l = Second_membre(vtx, elt, Lx, Ly, alpha)
    l = d@l

    u = spsolve(a, l)
    for i in range(n):
        x, y = vtx[i]
        if (x == 0.) or (y == 0.) or (x == Lx) or (y == Ly):
            u[i] = 0.
    return u

def Error(Lx: float, Ly: float, nb_sub_h: int, nb_sub_v: int, alpha: float) -> None:
    vtx, elt = generate_rectangle_mesh(Lx, Ly, nb_sub_h, nb_sub_v)
    m = assemble_mass_matrix(vtx, elt)
    n = vtx.shape[0]
    h = np.zeros((n,))
    err = np.zeros((n,))
    mu_test = np.zeros(n)
    d = generate_sparse_diagonal_matrix(vtx, Lx, Ly)
    I = sp.sparse.eye(n)

    for i in range(n):
        x = vtx[i][0]
        y = vtx[i][1]
        mu_test[i] = mu(x, y, Lx, Ly)
    a = global_matrix(vtx, elt, mu_test)
    a = d@a@d + I - d
    l = Second_membre(vtx, elt, Lx, Ly, alpha)
    l = d@l

    u = spsolve(a, l)
    for i in range(n):
        x, y = vtx[i]
        if (x == 0.) or (y == 0.) or (x == Lx) or (y == Ly):
            u[i] = 0.

    uex = [u_ex(v[0], v[1], 10, 6, 1) for v in vtx]
    delta = uex - u

    h[i] = 1 / n
    err[i] = np.sqrt(delta @ m @ delta) / np.sqrt(uex @ m @ uex)

    plt.figure("Error", figsize=(9, 6))
    plt.xlabel("h")
    plt.ylabel("error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(h, err)
    plt.show()

########################################### 5/ Tests ###################################################################

vtx, elt = generate_rectangle_mesh(10, 6, 10, 6)
ref_vtx, ref_elt =  geometric_refinement(vtx, elt, 3)
plot_mesh(ref_vtx, ref_elt)
plt.show()

u_h = solve(10, 6, 10, 6, 2/3)
plot_mesh(vtx, elt, u_h)
plt.title("Numerical solution u_h with alpha = 2/3")
plt.show()

u = [u_ex(v[0], v[1], 10, 6, 2/3) for v in vtx]
plot_mesh(vtx, elt, u)
plt.title("Exact solution u with alpha = 2/3")
plt.show()

plot_mesh(vtx, elt, u-u_h)
plt.title("Error u - u_h with alpha = 2/3")
plt.show()

u_h = solve(10, 6, 10, 6, 1)
plot_mesh(vtx, elt, u_h)
plt.title("Numerical solution u_h with alpha = 1")
plt.show()

u = [u_ex(v[0], v[1], 10, 6, 1) for v in vtx]
plot_mesh(vtx, elt, u)
plt.title("Exact solution u with alpha = 1")
plt.show()

plot_mesh(vtx, elt, u-u_h)
plt.title("Error u - u_h with alpha = 1")
plt.show()

n = vtx.shape[0]
mu_test = np.ones(n)
for i in range(n):
    x = vtx[i][0]
    y = vtx[i][1]
    mu_test[i] = mu(x, y, 10, 6)

print(mu_test, '\n')

K = assemble_stiffness_matrix(vtx, elt, mu_test)
M = assemble_mass_matrix(vtx, elt)

U = np.array([v[0] for v in vtx])
V = np.array([v[1] for v in vtx])
W = np.ones(n)

print("Tests on K :\n")
print("V@K@U = ", V@K@U)
print("U@K@U = ", U@K@U)
print("K@W = ", K@W)

print("\nTests on M :\n")
print("W@M@W = ", W@M@W)

print("\nTests Tests on functions:\n")
print(
u_ex(8, 3, 10, 6, 2), 
mu(0.3, 1.9, 10, 6), 
d_x_mu(0.3, 1.9, 10, 6), 
d_y_mu(0.3, 1.9, 10, 6),
d_x_u(8, 3, 10, 6, 2),
d_y_u(8, 3, 10, 6, 2),
d2_x_u(8, 3,10, 6, 2),
d2_y_u(8, 3, 10, 6, 2),
f(8, 3, 10, 6, 2))  