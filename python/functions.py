import scipy
import numpy as np


def integrate(H, dt, psi0):
    psi = psi0
    psi = scipy.linalg.expm(-1j * H * dt) @ psi
    return psi


def determine_jump(c_ops, psi, r2):
    jump = None
    p_tot = sum([np.linalg.norm(c_op @ psi) ** 2 for c_op in c_ops])
    p_it = 0
    for c_op in c_ops:
        prob = np.linalg.norm(c_op @ psi) ** 2
        p_it += prob / p_tot
        if p_it >= r2:
            jump = c_op
            break
    return jump


def montecarlo(H, c_ops, psi0, tlist):
    H_eff = H - 1j * sum([c.conj().T @ c for c in c_ops]) / 2

    dt = (max(tlist) - min(tlist)) / len(tlist)
    psi = psi0 / np.linalg.norm(psi0)
    psi_j = []
    psi_j.append(psi)

    r1 = np.random.rand()
    r2 = np.random.rand()
    for t_e in tlist:
        psi = integrate(H_eff, dt, psi)

        if np.linalg.norm(psi) ** 2 <= r1:
            jump = determine_jump(c_ops, psi, r2)
            psi = jump @ psi / np.linalg.norm(jump @ psi)
            r1 = np.random.rand()
            r2 = np.random.rand()

        psi_j.append(psi)

    psi_j.pop()
    return psi_j


def montecarlo_average(H, c_ops, psi0, tlist, ntraj, op):
    results = []
    for i in range(ntraj):
        results.append(montecarlo(H, c_ops, psi0, tlist))
    for i in range(len(results)):
        results[i] = [
            psi.conj().T @ op @ psi / np.linalg.norm(psi) ** 2 for psi in results[i]
        ]
    results = np.mean(results, axis=0)

    return results
