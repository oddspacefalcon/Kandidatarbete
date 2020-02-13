import numpy as np
system_size = 7
qubits = np.random.uniform(0, 1, size=(system_size, system_size))
p_error = 0.1
error = qubits > p_error
print(np.random.randint(3, size=(system_size, system_size)) + 1)
