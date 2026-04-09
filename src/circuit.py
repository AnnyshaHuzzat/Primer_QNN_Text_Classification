import pennylane as qml
import torch.nn as nn

n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

def variational_layer(weights):
    for i in range(n_qubits):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RZ(weights[i, 2], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.IQPEmbedding(inputs, wires=range(n_qubits))
    for w in weights:
        variational_layer(w)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}

def build_qlayer():
    return qml.qnn.TorchLayer(qnode, weight_shapes)
