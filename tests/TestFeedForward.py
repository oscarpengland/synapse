import numpy as np
import synapse

network = synapse.net.Sequential([
    synapse.net.FullyConnectedLayer(5, 8),
    synapse.net.Sigmoid(),

    synapse.net.FullyConnectedLayer(8, 10),
    synapse.net.Sigmoid(),
])

output = network.forward(np.array([[0.3], [0.1], [0.4], [0.5], [0.9]]))

print("Outputs =", output)