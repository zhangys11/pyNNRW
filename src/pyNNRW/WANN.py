'''
Weight Agnostic Neural Networks

To find architectures that can achieve much higher than chance accuracy using random weights
Use GA (or other random tool) to generate several candidates. -> choose the best one -> Finetune the last layer by MSE | or finetune the last few layers by back-propogation
Many neuroevolution algorithms have been defined. One common distinction is between algorithms that evolve only the strength of the connection weights for a fixed network topology (sometimes called conventional neuroevolution), as opposed to those that evolve both the topology of the network and its weights (called TWEANNs, for Topology and Weight Evolving Artificial Neural Network algorithms).
Most neural networks use gradient descent rather than neuroevolution. However, around 2017 researchers at Uber stated they had found that simple structural neuroevolution algorithms were competitive with sophisticated modern industry-standard gradient-descent deep learning algorithms, in part because neuroevolution was found to be less likely to get stuck in dead ends.
In direct encoding schemes the genotype directly maps to the phenotype. That is, every neuron and connection in the neural network is specified directly and explicitly in the genotype. In contrast, in indirect encoding schemes the genotype specifies indirectly how that network should be generated
'''

class WANN(object):
    pass # Research in future