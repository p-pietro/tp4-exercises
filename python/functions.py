import scipy
import numpy as np
from scipy.integrate import solve_ivp, ode


def schrodinger_eq(t, psi, H):
    dpsi_dt = -1j * H @ psi
    return dpsi_dt


class Integrator:
    def __init__(self):
        self.solver = ode(schrodinger_eq).set_integrator("zvode", method="adams")

    def integrate(self, H, psi0, t_span, t_eval):
        # Solve the Schrödinger equation using the ode function with the zvode solver
        if (t_eval < t_span[0]) or (t_eval > t_span[1]):
            raise ValueError("t_eval must be within t_span")

        self.solver.set_initial_value(psi0, t_span[0]).set_f_params(H)

        psi_t = psi0
        if self.solver.successful():
            self.solver.integrate(t_eval)
            psi_t = self.solver.y
        else:
            raise RuntimeError("ODE solver failed to integrate")

        return psi_t


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


def montecarlo(H, c_ops, psi0, tlist, seed=None):
    if seed is not None:
        np.random.seed(seed)

    integrator = Integrator()

    H_eff = H - 1j * sum([c.conj().T @ c for c in c_ops]) / 2

    dt = (max(tlist) - min(tlist)) / len(tlist)
    psi = psi0 / np.linalg.norm(psi0)
    psi_j = []
    psi_j.append(psi)
    t_prev = tlist[0]
    r1 = np.random.rand()
    r2 = np.random.rand()

    for t_idx in range(1, len(tlist)):
        t_span = (t_prev, tlist[t_idx])
        t_eval = tlist[t_idx]

        # Integrate using the ODE solver
        psi = integrator.integrate(H_eff, psi, t_span, t_eval)
        # psi = psi_t.y[:, -1]

        t_prev = tlist[t_idx]

        norm_sq = np.linalg.norm(psi) ** 2
        if norm_sq <= r1:
            jump = determine_jump(c_ops, psi, r2)
            psi = jump @ psi / np.linalg.norm(jump @ psi)
            r1 = np.random.rand()
            r2 = np.random.rand()

        psi_j.append(psi)

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
