#**********************************************#
# The file with operations which are applied   #
#   to the cosine matrices.                    #
#**********************************************#

import numpy as np
import numpy.linalg as lg

from scipy.spatial.distance import cdist

# coord - CA(CB) atoms coordinates (3xn)
# pmtrx - coords first difference (3x(n-1)) [normalized]
# lmtrx - non-normalized pmtrx
# tmtrx - (P @ P^t)^-1 @ P => T_p1 @ C_p1_p2 = P2

# BASIC OPERATIONS


def length(vecs: np.ndarray) -> np.ndarray:
    """Returns (1 x n) array of lengths of vecs (m x n) vectors."""
    return np.expand_dims(lg.norm(vecs, ord=2, axis=0), axis=0)


def diff(vecs: np.ndarray) -> np.ndarray:
    """Returns (m x n-1) array of first differences of vecs (m x n)."""
    return np.diff(vecs, axis=1)


def get_k(vecs: np.ndarray):
    """Get dimensionality of vecs."""
    return vecs.shape[0]


def normalize(vecs: np.ndarray) -> np.ndarray:
    """Normalizes vectors of vecs (3 x n)."""
    return vecs / np.repeat(length(vecs), repeats=get_k(vecs), axis=0)


def expand(vecs: np.ndarray) -> np.ndarray:
    """Adds first vector of vecs (m x n). Returns (m x n+1) array."""
    return np.column_stack([vecs, vecs[:, 0]])


def get_n(vecs: np.ndarray) -> np.ndarray:
    """Returns n of vecs (m x n)."""
    return vecs.shape[1]


def get_center(vecs: np.ndarray) -> np.ndarray:
    """Returns (m x 1) center of vecs (m x n)."""
    return np.expand_dims(np.mean(vecs, axis=1), axis=1)


def widen_upto(to_widen: np.ndarray, aim_array: np.ndarray) -> np.ndarray:
    return np.repeat(to_widen, repeats=get_n(aim_array), axis=1)


def add_column(mtrx: np.ndarray, column: np.ndarray) -> np.ndarray:
    """Adds column to a matrix."""
    return mtrx + widen_upto(column, mtrx)


def centralize(vecs: np.ndarray) -> np.ndarray:
    """Centralize (3 x n) vecs."""
    # return vecs - np.repeat(get_center(vecs), repeats=get_n(vecs), axis=1)
    return add_column(vecs, -get_center(vecs))


def get_pmtrx(coord: np.ndarray, need_expand=False) -> np.ndarray:
    """Constructs P-matrix from atom's coordinates."""
    return normalize(diff(expand(coord) if need_expand else coord))


get_pmtrx_from_coord = pmtrx_from_coord = get_pmtrx  # alias


def coord_from_pmtrx(pmtrx: np.ndarray, start_point=None) -> np.ndarray:
    """Construct coordinates matrix (3 x n) from (3 x n-1) P-matrix
                                           (or L-matrix, whatever).
       start_point - np.ndarray coordinates of starting point
       [by default - None, what corresponds np.zeros((3, 1))]."""
    if start_point is None:
        start_point = np.zeros((get_k(pmtrx), 1))
    return np.cumsum(np.column_stack([start_point, pmtrx]), 1)


get_coord_from_pmtrx = coord_from_pmtrx


def get_rmsd(coord1: np.ndarray, coord2: np.ndarray):
    """Calculates RMSD for two coordinate matrices of equal shape."""
    return np.sqrt(np.sum((coord1 - coord2) ** 2) / get_n(coord1))


# AMINO RESIDUES

# List of all residues.
all_res = ('UNK', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
           'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
           'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR',
           'TRP', 'TYR', 'VAL')

# List of all residues not including UNK
all_res_nu = all_res[1:]


def filter_amino(amino: list[str]) -> list[str]:
    """Filters all nonstandart residues as UNK."""
    return [res if res in all_res_nu else 'UNK' for res in amino]


# Mappers
map_res2idx = {x: xi for xi, x in enumerate(all_res)}
map_res2idx_nu = {x: xi for xi, x in enumerate(all_res_nu)}
map_idx2res = {y: x for x, y in map_res2idx.items()}
map_idx2res_nu = {y: x for x, y in map_res2idx_nu.items()}


def amino_list_to_indices(amino: list[str], consider_unk=True) -> np.ndarray:
    """Converts list of residues names into array of indices."""
    mapp = map_res2idx if consider_unk else map_res2idx_nu
    return np.array([mapp[res] if res in mapp.keys() else mapp['UNK']
                     for res in amino])


def amino_list_to_array(amino: list[str], consider_unk=True) -> np.ndarray:
    """Converts list of residues names into zero-ones array."""
    res = all_res if consider_unk else all_res_nu
    mtrx = np.eye(len(res))
    if not consider_unk:
        mtrx = np.row_stack([np.full((1, len(all_res_nu)), 0.05), mtrx])
    return mtrx[amino_list_to_indices(amino, True), :]


def amino_list_to_indices_stack(amino: list[str], consider_unk=True
                                ) -> np.ndarray:
    """Converts list of residues names into array of indices, stacked
                                                              (doubled)."""
    indices = amino_list_to_indices(amino, consider_unk)
    n = len(indices)
    return np.row_stack([indices[:n-1], indices[1:n]])


def amino_list_to_array_stack(amino: list[str], consider_unk=True
                              ) -> np.ndarray:
    """Converts list of residues names into zero-ones array, stacked
                                                             (double)."""
    array = amino_list_to_array(amino, consider_unk)
    n = array.shape[0]
    return np.column_stack([array[:n-1, :], array[1:n, :]])


# ROTATION MATICES


def euler_angles(alph, beth, gamm) -> np.ndarray:
    """Returns euler matrix for angles alpha, betha and gamma."""
    return np.array([[np.cos(alph), -np.sin(alph), 0],
                     [np.sin(alph),  np.cos(alph), 0],
                     [0,             0,            1]]) @ np.array(
                    [[1,             0,            0],
                     [0, np.cos(beth), -np.sin(beth)],
                     [0, np.sin(beth),  np.cos(beth)]]) @ np.array(
                    [[np.cos(gamm), 0, -np.sin(gamm)],
                     [0,            1,             0],
                     [np.sin(gamm), 0,  np.cos(gamm)]])

def euler_angles_degrees(alph, beth, gamm) -> np.ndarray:
    return euler_angles(alph * np.pi / 180, beth * np.pi / 180, gamm * np.pi / 180) 

def gen_q(alph, beth) -> np.ndarray:
    """Retruns Q matrix with 1-vector given in polar coordinates with alpha and
                                                               betha angles."""
    vec1 = np.array([np.cos(alph) * np.cos(beth),
                     np.cos(alph) * np.sin(beth),
                     np.sin(alph)])
    vec1 = np.expand_dims(vec1, axis=1)
    s_t = lg.qr(vec1, mode='complete')[0]
    return s_t @ np.diag([1, -1, -1]) @ s_t.T


def rand_gen_s() -> np.ndarray:
    """Returns random orthogonal matrix."""
    return euler_angles(np.random.rand() * 2 * np.pi,
                        np.random.rand() * 2 * np.pi,
                        np.random.rand() * 2 * np.pi)


def rand_gen_q() -> np.ndarray:
    """Returns random Q-matrix."""
    return gen_q(np.random.rand() * 2 * np.pi,
                 (np.random.rand() - 0.5) * np.pi)


# BASIC COSMAPS


def cosma_from_pmtrx(pmtrx1: np.ndarray, pmtrx2: np.ndarray) -> np.ndarray:
    """Construct COSMA from P-matrices."""
    return pmtrx1.T @ pmtrx2


def cosma_from_coords(coord1: np.ndarray, coord2: np.ndarray, need_expand=False
                      ) -> np.ndarray:
    """Construct COSMA from coordinates arrays."""
    pmtrx1, pmtrx2 = get_pmtrx(coord1,
                               need_expand), get_pmtrx(coord2, need_expand)
    return cosma_from_pmtrx(pmtrx1, pmtrx2)


get_cosma_from_pmtrx = cosma_from_pmtrx
get_cosma_from_coords = cosma_from_coords


def do_eigh(mtrx: np.ndarray, dim=3) -> tuple:
    """Convenient eigh call."""
    u, v = lg.eigh(mtrx)
    u = u[-dim:]
    v = v[:, -dim:]
    return (u, v)


# Extracting lmtrxs from cosma (errorneous)


def get_lmtrx_from_cosma_E(cosma_E: np.ndarray, dim=3) -> np.ndarray:
    """Extracts L-matrix, which should be equal to P-matrix for clear COSMA,
       but could require normalization for dirty COSMA
       dim - expected dimensionality of L-matrix (default 3)."""
    # n = cosma_E.shape[0]
    cosma_E = (cosma_E + cosma_E.T) / 2
    u, v = do_eigh(cosma_E, dim)
    return np.diag(np.sqrt(u)) @ np.transpose(v)


def get_pmtrx_from_cosma_E(cosma_E: np.ndarray, dim=3) -> np.ndarray:
    """ ... """
    return normalize(get_lmtrx_from_cosma_E(cosma_E, dim))


# !!!!!!!!!!!!!!! get_lmtrx_from_cosma_P1_P2

def get_tmtrx(pmtrx: np.ndarray) -> np.ndarray:
    """Calculates T-matrix for P-matrix:   T = (P @ P^t)^-1 @ P."""
    return lg.inv(pmtrx @ pmtrx.T) @ pmtrx


def orthogonalize(mtrx: np.ndarray, expected_det=None) -> np.ndarray:
    """Orthogonalize matrix with SVD [not good for zero-determinant matrices]
       expected_det is -1, 1 or None - determinant that matrix forced to have.
    """
    u, diag, vh = lg.svd(mtrx)
    diag = np.ravel(diag)
    if expected_det == 1 and lg.det(mtrx) < 0 or expected_det == -1 and lg.det(
            mtrx) > 0:
        least = np.argmin(np.abs(diag))
        diag[least] = -diag[least]
    return u @ np.diag(np.sign(diag)) @ vh


def get_s_from_cosma_p1_sp2(cosma_p1_sp2: np.ndarray, pmtrx1: np.ndarray,
                            pmtrx2: np.ndarray, expected_det=1) -> np.ndarray:
    """Calculates S (orthogonal) from P-matrices and COSMA P1, P2
                                                          (clear or dirty)."""
    return orthogonalize(get_tmtrx(pmtrx1) @ cosma_p1_sp2 @ get_tmtrx(
        pmtrx2).T, expected_det)


def orthogonalize_q(mtrx: np.ndarray, return_s=False):
    """Orthogonalize matrix to specter (-1, -1, 1) with eigenvector
                                                             decomposition.
       If return_s=True tuple (Q, S) will be returned, otherwise only Q."""
    w, v = lg.eigh((mtrx + mtrx.T) / 2)
    q = v @ np.diag([-1, -1, 1]) @ v.T
    return (q, v.T) if return_s else q


def get_q_from_cosma_p_qp(cosma_p_qp: np.ndarray, pmtrx: np.ndarray,
                          return_s=False) -> np.ndarray:
    """Calculates Q (orthogonal with specter [-1, -1, 1]) from P-matrix and
                                           COSMA P, QP (clear or dirty)."""
    tmtrx = get_tmtrx(pmtrx)
    return orthogonalize_q(tmtrx @ cosma_p_qp @ tmtrx.T, return_s)


# APPROXIMATIONS


def get_two_pmtrx_from_cosma_p1_p2(cosma_p1_p2: np.ndarray, dim=3) -> tuple:
    """Extract both P-matrices from COSMA_P1_P2 with respect to P1.
       That is APROXIMATION"""
    pmtrx1 = normalize(get_lmtrx_from_cosma_E(cosma_p1_p2 @ cosma_p1_p2.T,
                                              dim))
    pmtrx2 = normalize(get_tmtrx(pmtrx1) @ cosma_p1_p2)
    return (pmtrx1, pmtrx2)


approxy_p1_p2_from_cosma = get_two_pmtrx_from_cosma_p1_p2


def approxy_mult(cosma_p1_p2: np.ndarray, cosma_p2_p3: np.ndarray,
                 dim=3) -> np.ndarray:
    """ ... """
    n = get_n(cosma_p1_p2)
    return dim / n * cosma_p1_p2 @ cosma_p2_p3


def approxy_corr(cosma_p1_p2: np.ndarray, cosma_p2_p3: np.ndarray
                 ) -> np.ndarray:
    """ ... """
    m = cosma_p1_p2.shape[0]
    l = cosma_p2_p3.shape[1]
    return np.corrcoef(cosma_p1_p2, cosma_p2_p3.T)[:m, -l:]


# L-matrix normalization


# maybe better is just to use normalize...
# ---DO NOT USE (normalize instead)---
# USE WITH NORMALIZE


def get_pmtrx_from_lmtrx(lmtrx: np.ndarray, lens=None) -> np.ndarray:
    """Reconstructs P-matrix from L-matrix. May throw error if reconstruction
                                                           is impossible. [eww]
       lens - n x 1 array of desired lengths of P-matrix vectors, by default -
                                                                vector of ones.
       If L-matrix is dirty, P-matrix may require additional refinement."""
    mults = get_tmtrx(lmtrx ** 2) @ (np.ones((get_n(lmtrx), 1)
                                             ) if lens is None else lens ** 2)
    return np.diag(np.sqrt(mults[:, 0])) @ lmtrx
    # return np.diag(np.sqrt(get_tmtrx(lmtrx ** 2) @ (np.ones((get_n(lmtrx), 1)
    # ) if lens is None else lens ** 2))[:, 0]) @ lmtrx


def get_two_pmtrx_from_cosma_p1_p2_correct(cosma_p1_p2: np.ndarray,
                                           dim=3) -> tuple:
    """Extract both P-matrices from COSMA_P1_P2 with respect to P1.
       Should be mathematically correct for clean matrices."""
    P1t_Hp1_1_V = do_eigh(cosma_p1_p2 @ cosma_p1_p2.T, dim)[1]
    P2t_Hp2_1_VA = do_eigh(cosma_p1_p2.T @ cosma_p1_p2, dim)[1]
    Vt_P2 = P1t_Hp1_1_V.T @ cosma_p1_p2
    AV_1_P1 = (cosma_p1_p2 @ P2t_Hp2_1_VA).T
    A_1 = get_tmtrx(AV_1_P1) @ cosma_p1_p2 @ get_tmtrx(Vt_P2).T
    B_St_V = do_eigh(Vt_P2 @ Vt_P2.T)[1].T  # why is it E? whatever
    B_St_P1 = B_St_V @ A_1 @ AV_1_P1
    p1 = normalize(get_pmtrx_from_lmtrx(B_St_P1))
    p2 = normalize(get_tmtrx(p1) @ cosma_p1_p2)
    return (p1, p2)


p1_p2_from_cosma = get_two_pmtrx_from_cosma_p1_p2_correct

# Alignment of structures


def align_lmtrx_to_coord(lmtrx: np.ndarray, coord_aim: np.ndarray,
                         verbose=False, expected_det=None):
    """Aligns structure, written as L-matrix, to known coordinates via
       rotation/reflection (controlled with expected_det) and shift.
       If verbose=False returns only aligned coordinates, otherwise returns
       tuple (coord_aligned, q_mtrx, shift_vec)."""
    lmtrx_aim = diff(coord_aim)
    q = orthogonalize(lmtrx_aim @ get_tmtrx(lmtrx).T, expected_det)
    coord_aligned = coord_from_pmtrx(q @ lmtrx)
    shift_vec = get_center(coord_aim - coord_aligned)
    # coord_aligned = coord_aligned + np.repeat(shift_vec,
    # repeats=get_n(coord_aligned), axis=1)
    coord_aligned = add_column(coord_aligned, shift_vec)
    return (coord_aligned, q, shift_vec) if verbose else coord_aligned


# RANDOM PROTEIN
silent = False

try:
    import pickle as pkl
    with open('./probs.pkl', 'rb') as file:
        probs = list(pkl.load(file)[1].values())

    def rand_prot_gen(n):
        """Generates random protein sequence (emperical frequences) with length
                                                                       of n."""
        return [map_idx2res_nu[np.random.choice(len(probs), p=probs)]
                for i in range(n)]
except Exception:
    if not silent:
        print("Couldn't load './probs.pkl'.")


# #########################
# PROCCESSING INTERACTIONS#
# #########################

def get_cv(coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
    """Returns central vector of first protein towards second protein."""
    return normalize(get_center(coord2) - get_center(coord1))


def cosine(vec1: np.ndarray, vec2: np.ndarray):
    return np.sum(vec1 * vec2) / lg.norm(vec1) / lg.norm(vec2)


def cos_to_angl(cos):
    return np.arccos(cos) * 180 / np.pi


def get_dst(coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
    """Returns distances matrix."""
    # return np.sqrt(np.array([[np.sum((coord1[:, i] - coord2[:, j]) ** 2)
    # for j in range(get_n(coord2))] for i in range(get_n(coord1))]))
    return cdist(coord1.T, coord2.T)


def bump(coord1: np.ndarray, coord2: np.ndarray, border0, border1, perc, step,
         cv=None) -> np.ndarray:
    """Bumping second protein on the first protein, with:
           border0 - minimal allowed distance between atoms;
           border1 - main distance border (for at least perc pairs);
           perc - amount of pairs, allowed to be less than border1 but more
                                                              than border0;
           step - shift length in angstroms;
           cv - center vector (shifting directions),
                by default would be computed from given coordinates."""
    if cv is None:
        cv = get_cv(coord1, coord2)
    cv = cv * step
    center1 = get_center(coord1)
    coord2 = add_column(centralize(coord2), center1)
    for s in range(1000000):
        dst = get_dst(coord1, coord2)
        if np.all(dst >= border0) and np.mean(dst >= border1) >= perc:
            break
        coord2 = add_column(coord2, cv)
    return coord2


def get_cuts(coord: np.ndarray, border, return_sum=True):
    cuts = length(diff(coord)) > 4.5
    return np.sum(cuts) if return_sum else cuts


def check_homo(amino1: list[str], amino2: list[str]) -> bool:  # WEAK
    """Checking homodimerity by checking amino sequences [neglecting cuts]."""
    return len(amino1) == len(amino2) and np.all([amino1[i] == amino2[i]
                                                  for i in range(len(amino1))])


def check_connection(coord1: np.ndarray, coord2: np.ndarray, border0, border1,
                     perc, step, rmsd_border) -> bool:
    """Checks, is two given protens chains likely to be connected. [...]"""
    if np.any(get_dst(coord1, coord2) < border0):  # hmm...
        return False
    coord2_ = bump(coord1, coord2, border0, border1, perc, step)
    return get_rmsd(coord2, coord2_) < rmsd_border


def check_default(coord1: np.ndarray, coord2: np.ndarray) -> bool:
    return check_connection(coord1, coord2, 3.4, 3.6, 0.1, 0.1, 3)

def bump_default(coord1: np.ndarray, coord2: np.ndarray, cv: np.ndarray
                 ) -> np.ndarray:
    return bump(coord1, coord2, 3.4, 3.6, 0.1, 0.1, cv)
    # return old_bump_default(coord1, coord2, cv)


# smth about interfaces... to think about...

def refine_cv_matrix(cv_matrix: np.ndarray, pmtrx: np.ndarray, verbose=False):
    """Simple refinement of CV (right) matrix into singe 3d CV vector.
       In verbose mode returns dict containing 3d vector,
       m-d vector and deviation"""
    cv_3d = normalize(get_tmtrx(pmtrx) @ get_center(cv_matrix))
    if verbose:
        cv_md = get_cosma_from_pmtrx(pmtrx, cv_3d)
        dev = np.sqrt(np.mean(add_column(cv_matrix, -cv_md) ** 2))
        return {'cv_3d': cv_3d, 'cv_md': cv_md, 'dev': dev}  # cv_mtrx?
    else:
        return cv_3d


def compare_models(coord1: np.ndarray, coord2_true: np.ndarray,
                   coord2_pred: np.ndarray, homo=False) -> dict:
    """Comparing true model with predicted model with
       variuos metrics."""
    rmsd = get_rmsd(coord2_true, coord2_pred)
    cv_true = get_cv(coord1, coord2_true)
    cv_pred = get_cv(coord1, coord2_pred)
    cv_cos = cosine(cv_true, cv_pred)
    cv_angl = cos_to_angl(cv_cos)

    pmtrx2_true = get_pmtrx(coord2_true)
    pmtrx2_pred = get_pmtrx(coord2_pred)
    s = get_tmtrx(pmtrx2_true) @ pmtrx2_pred.T
    s_norm = lg.norm(s - np.identity(3))

    result = {'rmsd': rmsd, 'cv_cos': cv_cos, 'cv_angl': cv_angl,
              's_norm': s_norm}

    if homo:
        tmtrx1 = get_tmtrx(get_pmtrx(coord1))
        q_true, vT_true = orthogonalize_q(tmtrx1 @ pmtrx2_true.T, True)
        q_pred, vT_pred = orthogonalize_q(tmtrx1 @ pmtrx2_pred.T, True)
        v1_cos = np.abs(cosine(vT_true[2, :], vT_pred[2, :]))
        v1_angl = cos_to_angl(v1_cos)
        q_norm = lg.norm(q_true - q_pred)

        result = {**result, 'v1_cos': v1_cos, 'v1_angl': v1_angl,
                  'q_norm': q_norm}
    return result


# Hmmm, Hmmmm, Hmmmm...

def full_model(coord1: np.ndarray, coord2: np.ndarray, cosma: np.ndarray,
               cv1: np.ndarray, cv2: np.ndarray, homo: bool, expanded=False
               ) -> dict:
    """Full model proccessing."""

    cv_mtrx = get_n(cv1) != 1   # hehehe

    if homo:
        pmtrx = pmtrx_from_coord(coord1, expanded)
        tmtrx = get_tmtrx(pmtrx)
        q, v = get_q_from_cosma_p_qp(cosma, pmtrx, True)
        v1 = np.expand_dims(v[2, :], 1)
        if cv_mtrx:             # using cv_matrix
            cv_3d = refine_cv_matrix(np.column_stack([cv1, cv2.T]), pmtrx)
        else:                   # using linear cv
            cv1_3d = normalize(tmtrx @ cv1)
            cv2_3d = normalize(tmtrx @ cv2)
            cv_3d = (cv1_3d + cv2_3d) / 2  # can be without / 2...
        cv_3d = normalize(cv_3d - np.sum(v1 * cv_3d) * v1)
        coord_pred = bump_default(coord1, q @ coord1, cv_3d)

        if cv_mtrx:
            cv1_dev = np.sqrt(np.mean(add_column(cv1, -pmtrx.T @ cv_3d) ** 2))
            cv2_dev = np.sqrt(np.mean(add_column(cv2.T,
                                                 -pmtrx.T @ cv_3d) ** 2))
        else:
            cv1_dev = np.sqrt(np.mean((cv1 - pmtrx.T @ cv_3d) ** 2))
            cv2_dev = np.sqrt(np.mean((cv2 - pmtrx.T @ cv_3d) ** 2))
        cosma_dev = np.sqrt(np.mean((get_cosma_from_pmtrx(
            pmtrx, q @ pmtrx) - cosma) ** 2))

    else:
        pmtrx1 = pmtrx_from_coord(coord1, expanded)
        pmtrx2 = pmtrx_from_coord(coord2, expanded)
        s = get_s_from_cosma_p1_sp2(cosma, pmtrx1, pmtrx2)
        if cv_mtrx:
            cv1_3d = refine_cv_matrix(cv1, pmtrx1)
            cv2_3d = refine_cv_matrix(cv2.T, pmtrx2)
        else:
            cv1_3d = normalize(get_tmtrx(pmtrx1) @ cv1)
            cv2_3d = normalize(get_tmtrx(pmtrx2) @ cv2)
        cv_3d = normalize(cv1_3d - s @ cv2_3d)
        coord_pred = bump_default(coord1, s @ coord2, cv_3d)

        if cv_mtrx:
            cv1_dev = np.sqrt(np.mean(add_column(cv1, -pmtrx1.T @ cv_3d) ** 2))
            cv2_dev = np.sqrt(np.mean(
                add_column(cv2.T, -pmtrx2.T @ s.T @ cv_3d) ** 2))
        else:
            cv1_dev = np.sqrt(np.mean((cv1 - pmtrx1.T @ cv_3d) ** 2))
            cv2_dev = np.sqrt(np.mean((cv2 + pmtrx2.T @ s.T @ cv_3d) ** 2))
        cosma_dev = np.sqrt(np.mean((get_cosma_from_pmtrx(
            pmtrx1, s @ pmtrx2) - cosma) ** 2))

    return {'coord': (coord1, coord_pred), 'cv_dev': (cv1_dev, cv2_dev),
            'cosma_dev': cosma_dev}
