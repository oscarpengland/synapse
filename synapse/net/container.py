class Sequential:
    def __init__(self, layers: list):
        """
        Define a sequential network. architecture is a list of the various layers. e.g.
        new_network = nn.Sequential([
            ...
        ])
        """
        self.layers = layers
        self.optimizer = None

    def forward(self, inputs):
        """
        Perform forward pass over entire network.
        """
        a = inputs
        for layer in self.layers:
            a = layer.forward(a)
        return a