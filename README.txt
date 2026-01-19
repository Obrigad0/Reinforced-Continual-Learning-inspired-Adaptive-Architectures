Reinforced Continual Learning-inspired Adaptive Architectures

This project is inspired by Reinforced Continual Learning (RCL), where a reinforcement learning
(RL) controller learns to adapt a neural network’s architecture over a sequence of tasks in order
to mitigate catastrophic forgetting while controlling model complexity. In the original work,
the base learner is a supervised model, and a controller network chooses architectural actions
(e.g., reusing, expanding, or freezing parts of the network) for each new task. The controller
is trained with a policy-gradient algorithm using a reward that trades off accuracy on both old
and new tasks against the growth in the number of parameters. The key idea is to treat archi
tecture selection itself as a sequential decision-making problem, enabling data-driven discovery
of task-dependent structures that support knowledge retention and efficient capacity allocation.
As possible extensions, the project will mainly focus on exploring variations of the original
RCL framework. In particular, one could try different choices for the controller model(for
example a deeper RNN controller, a small Transformer-style sequence model, or small architec
tural changes such as different hidden sizes) and for the task network (for example alternative
CNN backbones with different depth or width, using a slightly simpler network vs. a slightly
larger one to study the effect of capacity).

The same approach could be tested on a additional task sequences or datasets, chosen among
standard continual learning benchmarks, to check whether the behaviour observed in the original
paper is stable when the tasks or data distribution are modified in simple, controlled ways.

Reference
• Ju Xu and Zhanxing Zhu. Reinforced Continual Learning. 2018