import numpy as np
from functions import montecarlo, montecarlo_average
from matplotlib import pyplot as plt
import time

# parameters for damped oscillations
N = 2  # only two fock states
wc = 1.0e-3 * 2 * np.pi  # cavity frequency
wa = wc  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.005  # cavity dissipation rate
gamma = 0.05  # atom dissipation rate
n_th_a = 0.1  # temperature in frequency units

psi0 = np.kron((1, 0), (0, 1)).astype(complex)
a = np.zeros((N, N))
for n in range(1, N):
    a[n - 1, n] = np.sqrt(n)
a = np.kron(a, np.eye(2))
a_dag = np.zeros((N, N))
for n in range(N - 1):
    a_dag[n + 1, n] = np.sqrt(n + 1)
a_dag = np.kron(a_dag, np.eye(2))

sm = np.zeros((2, 2))
sm[0, 1] = 1
sm = np.kron(np.eye(N), sm)
sp = np.zeros((2, 2))
sp[1, 0] = 1
sp = np.kron(np.eye(N), sp)
sz = np.zeros((2, 2))
sz[0, 0] = 1
sz[1, 1] = -1
sz = np.kron(np.eye(N), sz)

H = wc * a_dag @ a + wa / 2 * sz + g * (a_dag @ sm + a @ sp)

c_op_list = []
c_op_list.append(np.sqrt(kappa * (1 + n_th_a)) * a)
c_op_list.append(np.sqrt(kappa * n_th_a) * a_dag)
c_op_list.append(np.sqrt(gamma * (1 + n_th_a)) * sm)
c_op_list.append(np.sqrt(gamma * n_th_a) * sp)

times = np.linspace(0, 250, 100000)

start_time = time.time()
result_ode = montecarlo(H, c_op_list, psi0, times, 42)

end_time = time.time()
print(f"Time taken using zvode: {end_time - start_time} seconds")

start_time = time.time()
result_exp = montecarlo(H, c_op_list, psi0, times, 42, False)

end_time = time.time()
print(f"Time taken using matrix exponential: {end_time - start_time} seconds")

P_evolution_ode = [np.linalg.norm(psi) ** 2 for psi in result_ode]
P_evolution_exp = [np.linalg.norm(psi) ** 2 for psi in result_exp]


plt.figure(figsize=(10, 6))

plt.plot(times, P_evolution_ode, label="zvode")
plt.plot(times, P_evolution_exp, label="matrix exponential")
plt.xlabel("Time")
plt.ylabel("norm")
plt.title("Norm evolution over time")
plt.legend()
plt.grid(True)
plt.show()
