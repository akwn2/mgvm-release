import numpy as np
import scipy.special as ss
import scipy.linalg as la
from scipy.optimize import fsolve
from utils.special import gve


def log_z_vm(k):
    return np.log(2 * np.pi * ss.ive(0, k)) + k


def obj_inv_ive(z, nu, x):
    return np.log(ss.ive(nu, z)) + z - x


def grad_inv_ive(z, nu, x):
    if nu != 0:
        idx = np.abs(z) < 1E-10
        z[idx] = np.sign(z[idx]) * 1.E-10  # safeguard against division by zero
        return 0.5 * (ss.ive(nu + 1, z) + ss.ive(nu - 1, z)) / ss.ive(nu, z)
    else:
        return ss.ive(1, z) / ss.ive(0, z)


def inv_log_iv(nu, x, z0):
    # Finds the inverse of the log of a modified bessel function of order nu by newton method
    ans = np.zeros_like(x)
    for ii in xrange(0, np.alen(x)):
        ans[ii] = fsolve(func=obj_inv_ive, x0=z0[ii], args=(nu, x[ii]), fprime=grad_inv_ive)
    return ans


def moments_to_vm(log_mom_0, mom_1):

    m1 = np.angle(mom_1)

    x = log_mom_0 - np.log(2. * np.pi)
    z0 = np.ones_like(x) * 1000.
    k1 = inv_log_iv(nu=0, x=x, z0=z0)

    return m1, np.abs(k1)


def gvm_moments(m1, m2, k1, k2):
    mom0 = gve(0, m1, m2, k1, k2).real
    mom1 = gve(1, m1, m2, k1, k2) / mom0

    return np.log(mom0) + k1 + k2, mom1


def bgvm_moments(mu, kappa, mat_kin):

    res = 1000
    th0, th1 = np.meshgrid(np.linspace(-np.pi, np.pi, num=res, endpoint=False),
                           np.linspace(-np.pi, np.pi, num=res, endpoint=False))

    delta_sq = (th0[0, 1] - th0[0, 0]) ** 2

    weight = np.sum(kappa) + np.sum(mat_kin)

    energy = np.exp(kappa[0] * np.cos(th0 - mu[0]) + kappa[1] * np.cos(th1 - mu[1]) - 0.5 * (
        np.cos(th0) ** 2 * mat_kin[0, 0] + 2 * np.cos(th0) * np.cos(th1) * mat_kin[0, 1] +
        np.cos(th1) ** 2 * mat_kin[1, 1] +
        np.sin(th1) ** 2 * mat_kin[2, 2] + 2 * np.sin(th0) * np.sin(th1) * mat_kin[2, 3] +
        np.sin(th1) ** 2 * mat_kin[3, 3] +
        2 * np.cos(th0) * np.sin(th0) * mat_kin[0, 2] + 2 * np.cos(th0) * np.sin(th1) * mat_kin[0, 3] +
        2 * np.cos(th1) * np.sin(th1) * mat_kin[1, 3] + 2 * np.sin(th0) * np.cos(th1) * mat_kin[2, 1] - weight
    ))

    # Trapezoidal rule form
    energy[1:-2, :] *= 2.
    energy[:, 1:-2] *= 2.

    mom_1_th0 = energy * np.exp(1.j * th0)

    mom_1_th1 = energy * np.exp(1.j * th1)

    z = np.sum(energy, keepdims=True) * 0.25 * delta_sq

    mom_1 = np.array([np.sum(mom_1_th0) * 0.25 * delta_sq / z, np.sum(mom_1_th1) * 0.25 * delta_sq / z])

    return np.log(z) + weight, mom_1


def extract_submatrix(node, neig, mat_kin):

    nn = mat_kin.shape[0] / 2

    sub_mat_kin = np.array(
        [[mat_kin[node, node], mat_kin[node, neig], mat_kin[node, node + nn], mat_kin[node, neig + nn]],
         [mat_kin[neig, node], mat_kin[neig, neig], mat_kin[neig, node + nn], mat_kin[neig, neig + nn]],
         [mat_kin[node + nn, node], mat_kin[node + nn, neig], mat_kin[node + nn, node + nn], mat_kin[node + nn, neig + nn]],
         [mat_kin[neig + nn, node], mat_kin[neig + nn, neig], mat_kin[neig + nn, node + nn], mat_kin[neig + nn, neig + nn]]]
    )

    return sub_mat_kin


def run_ep(config):

    # Load parameters
    # n_data = config['N_data']
    c_data = config['c_data']
    s_data = config['s_data']
    # c_2data = config['c_2data']
    # s_2data = config['s_2data']
    mat_kin = config['Kinv']  # not ideal, but this way it's all consistent.
    k1 = config['k1']
    # k2 = config['k2']
    damping = config['damping']
    max_sweeps = config['max_sweeps']
    tol = config['tol']

    cpx_data_1 = k1 * c_data + 1.j * s_data
    # cpx_data_2 = k2 * c_2data + 1.j * s_2data

    n_vars = mat_kin.shape[0] / 2

    # Variable pairs:
    # Here we list all variable nodes and their neighbours without repetition forming a pair that corresponds to the
    # true factor i.e. tf(node, neig) which is approximated by the product of approximate factors af(node) * af(neig)
    pairs = list()
    for node in xrange(0, n_vars):
        for neig in xrange(node + 1, n_vars):
            pairs.append([node, neig])

    pairs = np.asarray(pairs)
    n_truf = pairs.shape[0]

    m1_f = np.random.rand(n_truf, 2) * 2. * np.pi - np.pi
    k1_f = np.random.rand(n_truf, 2) * 10.

    # Main EP loop:
    converged = False
    swept = 0
    res = list()
    while swept < max_sweeps and not converged:
        k1_f_old = np.copy(k1_f)
        m1_f_old = np.copy(m1_f)

        for nf in xrange(0, n_truf):

            print 'Sweep: ' + str(swept) + ' -> updating node ' + str(nf) + ' of ' + str(n_truf) + '.'

            # First, get the identifier of the node and the neig factors
            node = pairs[nf, 0]
            neig = pairs[nf, 1]

            # Now we will form the cavity distribution
            # ------------------------------------------------------------------------

            # 1. find the approximating factors who are functions of variable node
            all_node_on_0 = pairs[:, 0] == node
            all_node_on_1 = pairs[:, 1] == node
            all_node_on_0[nf] = False  # remove the nf index to avoid double counting

            # 2. find the approximating factors who are functions of variable neig
            all_neig_on_0 = pairs[:, 0] == neig
            all_neig_on_1 = pairs[:, 1] == neig
            all_neig_on_1[nf] = False  # remove the nf index to avoid double counting

            # 3. Agglomerate approximating factors together to calculate the moments
            cpx_cav_node = np.sum(k1_f[all_node_on_0, 0] * np.exp(1.j * m1_f[all_node_on_0, 0])) +\
                           np.sum(k1_f[all_node_on_1, 1] * np.exp(1.j * m1_f[all_node_on_1, 1]))

            k1_cav_node = np.abs(cpx_cav_node)
            m1_cav_node = np.angle(cpx_cav_node)

            cpx_cav_neig = np.sum(k1_f[all_neig_on_0, 0] * np.exp(1.j * m1_f[all_neig_on_0, 0])) +\
                           np.sum(k1_f[all_neig_on_1, 1] * np.exp(1.j * m1_f[all_neig_on_1, 1]))

            k1_cav_neig = np.abs(cpx_cav_neig)
            m1_cav_neig = np.angle(cpx_cav_neig)

            m1_cavity = np.array([m1_cav_node, m1_cav_neig])
            k1_cavity = np.array([k1_cav_node, k1_cav_neig])

            # Get moments of the true distribution
            # ------------------------------------------------------------------------

            sub_mat_kin = extract_submatrix(node, neig, mat_kin)
            log_z_true, mom_1_true = bgvm_moments(m1_cavity, k1_cavity, sub_mat_kin)

            # Update the parameters
            # ------------------------------------------------------------------------

            # Calculate the effect of the neig factor to remove it from when moment matching node variable
            cpx_neig_fx = k1_cav_neig * np.exp(1.j * m1_cav_neig) + k1_f[nf, 1] * np.exp(1.j * m1_f[nf, 1])
            log_z_neig = log_z_vm(np.abs(cpx_neig_fx))

            # Get new node factor parameters
            log_z_update = log_z_true - log_z_neig
            new_m1_node, new_k1_node = moments_to_vm(log_mom_0=log_z_update, mom_1=mom_1_true[0])

            # Calculate the effect of the node factor to remove it from when moment matching neig variable
            cpx_node_fx = k1_cav_node * np.exp(1.j * m1_cav_node) + k1_f[nf, 0] * np.exp(1.j * m1_f[nf, 0])
            log_z_node = log_z_vm(np.abs(cpx_node_fx))

            # Get new neig factor parameters
            log_z_update = log_z_true - log_z_node
            new_m1_neig, new_k1_neig = moments_to_vm(log_mom_0=log_z_update, mom_1=mom_1_true[1])

            # Add damping on the update
            k1_f[nf, 0] = (1. - damping) * new_k1_node + damping * k1_f[nf, 0]
            m1_f[nf, 0] = (1. - damping) * new_m1_node + damping * m1_f[nf, 0]
            k1_f[nf, 1] = (1. - damping) * new_k1_neig + damping * k1_f[nf, 1]
            m1_f[nf, 1] = (1. - damping) * new_m1_neig + damping * m1_f[nf, 1]

            print '    Factors (' + str(node) + ', ' + str(neig) + ') values:'
            print '\n'
            print '        m1_old' + str(node) + ': ' + str(m1_f_old[nf, 0])
            print '        m1_new' + str(node) + ': ' + str(m1_f[nf, 0])
            print '        k1_old' + str(node) + ': ' + str(k1_f_old[nf, 0])
            print '        k1_new' + str(node) + ': ' + str(k1_f[nf, 0])
            print '\n'
            print '        m1_old' + str(neig) + ': ' + str(m1_f_old[nf, 1])
            print '        m1_new' + str(neig) + ': ' + str(m1_f[nf, 1])
            print '        k1_old' + str(neig) + ': ' + str(k1_f_old[nf, 1])
            print '        k1_new' + str(neig) + ': ' + str(k1_f[nf, 1])
            print '\n'

        res.append(la.norm(k1_f * np.cos(m1_f) - k1_f_old * np.cos(m1_f_old)) +
                   la.norm(k1_f * np.sin(m1_f) - k1_f_old * np.sin(m1_f_old)))
        converged = res[-1] < tol
        swept += 1

    if converged:
        print 'Converged!'
    else:
        print 'Convergence not attained'

    return k1_f, m1_f, res, swept, converged

