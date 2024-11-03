import scipy
import numpy as np
from scipy.integrate import solve_ivp, ode


def schrodinger_eq(t, psi, H):
    dpsi_dt = -1j * H @ psi
    return dpsi_dt


class Integrator:
    def __init__(self, use_ode=True):
        self.solver = ode(schrodinger_eq).set_integrator(
            "zvode", method="adams", atol=1e-8, rtol=1e-6, first_step=1e-8
        )
        self.use_ode = use_ode

    def integrate(self, H, psi0, t_span, t_eval):
        if self.use_ode:
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
        else:
            # Integrate using matrix exponential
            dt = t_span[1] - t_span[0]
            psi_t = scipy.linalg.expm(-1j * H * dt) @ psi0

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


def montecarlo(H, c_ops, psi0, tlist, seed=None, use_ode=True):
    if seed is not None:
        np.random.seed(seed)

    integrator = Integrator(use_ode=use_ode)

    H_eff = H - 1j * sum([c.conj().T @ c for c in c_ops]) / 2

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

        norm_sq = np.linalg.norm(psi) ** 2
        if norm_sq <= r1:
            jump = determine_jump(c_ops, psi, r2)
            psi = jump @ psi / np.linalg.norm(jump @ psi)
            r1 = np.random.rand()
            r2 = np.random.rand()

        t_prev = t_eval
        psi_j.append(psi)

    return psi_j


def montecarlo_average(H, c_ops, psi0, tlist, ntraj, op, seed=None, use_ode=True):
    results = []
    for i in range(ntraj):
        results.append(
            montecarlo(H, c_ops, psi0, tlist, use_ode=use_ode, seed=seed + i)
        )
    for i in range(len(results)):
        results[i] = [
            psi.conj().T @ op @ psi / np.linalg.norm(psi) ** 2 for psi in results[i]
        ]
    results = np.mean(results, axis=0)

    return results
