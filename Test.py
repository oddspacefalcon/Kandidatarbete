from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
from src.util import Perspective, Action

for i in range(100):
    device = 'cpu'
    system_size = 7
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)
    model = ResNet18()
    args = {'cpuct': 0.5, 'num_simulations':30, 'grid_shift': system_size//2}

    mcts = MCTS(toric_code, model, device, args)
    pi, actions = mcts.get_probs_actions()

    print('antal perspektiv:', len(pi))
    print('pi:', pi)
    print('----------------------------------')