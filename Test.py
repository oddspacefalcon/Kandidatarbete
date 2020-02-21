from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS

for i in range(20):
    device = 'cpu'
    system_size = 7
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)
    model = ResNet18()
    args = {'cpuct': 0.5, 'num_simulations':20, 'grid_shift': system_size//2}

    mcts = MCTS(toric_code, model, device, args)
    pi, z = mcts.get_probs_v()

    print('antal perspektiv:', len(pi))
    print('pi:', pi, '\nz:', z)
    print('----------------------------------')