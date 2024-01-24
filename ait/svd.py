import numpy as np

def isvd(U, S, V, n_comp):
    return (U[:, :n_comp] * S[:n_comp]) @ V[:n_comp, :]

def svd_img(img):
    U_list = []
    S_list = []
    V_list = []
    for i in range(3):
        U, S, V = np.linalg.svd(img[:, :, i])
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    # to arr
    U = np.stack(U_list, axis=-1)
    S = np.stack(S_list, axis=-1)
    V = np.stack(V_list, axis=-1)
    return U, S, V

def isvd_img(U, S, V, n_cmop):
    return np.stack(
        [isvd(U[:, :, i], S[:, i], V[:, :, i], n_cmop) for i in range(3)],
        axis=-1)
