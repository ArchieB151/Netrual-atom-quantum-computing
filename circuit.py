import numpy as np
from qutip_qip.device import OptPulseProcessor, LinearSpinChain, SpinChainModel, SCQubits
from qutip_qip.circuit import QubitCircuit
from qutip import sigmaz, sigmax, identity, tensor, basis, ptrace

qc = QubitCircuit(N=3)
qc.add_gate("X", targets=2)
qc.add_gate("SNOT", targets=0)
qc.add_gate("SNOT", targets=1)
qc.add_gate("SNOT", targets=2)

# function f(x)
qc.add_gate("CNOT", controls=0, targets=2)
qc.add_gate("CNOT", controls=1, targets=2)

qc.add_gate("SNOT", targets=0)
qc.add_gate("SNOT", targets=1)

processor = LinearSpinChain(3)
processor.load_circuit(qc);

processor.plot_pulses(title="Control pulse of Spin chain", figsize=(8, 4), dpi=100);
