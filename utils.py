import random
from typing import List
import matplotlib.pyplot as plt
import torch
from dqn import DQN



def greedy_action(dqn: DQN, state: torch.Tensor) -> int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))


def epsilon_greedy(epsilon: float, dqn: DQN, state: torch.Tensor) -> int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p > epsilon:
        return greedy_act
    else:
        return random.randint(0, num_actions - 1)


def update_target(target_dqn: DQN, policy_dqn: DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())


def loss(policy_dqn: DQN, target_dqn: DQN,
         states: torch.Tensor, actions: torch.Tensor,
         rewards: torch.Tensor, next_states: torch.Tensor,
         dones: torch.Tensor, ddqn=False) -> torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """
    if ddqn:
        best_action_index = torch.argmax(policy_dqn(states), 1).reshape(-1, 1)
        bellman_targets = (~dones).reshape(-1) * (target_dqn(next_states).gather(1, best_action_index)).reshape(-1) + rewards.reshape(-1)
    else:
        bellman_targets = (~dones).reshape(-1) * (target_dqn(next_states)).max(
            1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets) ** 2).mean()


def create_uniform_neurons_per_layer_list(no_of_input_neurons: int,
                                          no_of_output_neurons: int,
                                          no_of_layers: int,
                                          no_of_neurons_per_hidden_layer:
                                          int = 0) \
                                        -> List[int]:
    """Create a list of the number of (input, output) neurons per layer."""
    if no_of_layers == 1:
        return [no_of_input_neurons, no_of_output_neurons]
    neurons_per_layer = []
    for i in range(no_of_layers - 1):
        if i == 0:
            neurons_per_layer.append(no_of_input_neurons)
            neurons_per_layer.append(no_of_neurons_per_hidden_layer)
        elif i == no_of_layers - 2:
            neurons_per_layer.append(no_of_neurons_per_hidden_layer)
            neurons_per_layer.append(no_of_output_neurons)
        else:
            neurons_per_layer.append(no_of_neurons_per_hidden_layer)
    return neurons_per_layer


def visualise_slice(network, cart_velocity, visualise_q):
    policy_net = network   # randomly initialised, replace with your trained DQN
    q = visualise_q    # whether q values or greedy policy is visualised

    angle_range = .2095 # you may modify this range
    omega_range = 2     # you may modify this range

    angle_samples = 100
    omega_samples = 100
    angles = torch.linspace(angle_range, -angle_range, angle_samples)
    omegas = torch.linspace(-omega_range, omega_range, omega_samples)

    greedy_q_array = torch.zeros((angle_samples, omega_samples))
    policy_array = torch.zeros((angle_samples, omega_samples))
    for i, angle in enumerate(angles):
        for j, omega in enumerate(omegas):
            state = torch.tensor([0., cart_velocity, angle, omega])
            with torch.no_grad():
                q_vals = policy_net(state)
                greedy_action = q_vals.argmax()
                greedy_q_array[i, j] = q_vals[greedy_action]
                policy_array[i, j] = greedy_action
    if q:
        plt.contourf(angles, omegas, greedy_q_array.T, cmap='cividis', levels=100)
        name = f'Greedy Q-Values of the DQN, Cart Velocity = {cart_velocity}'
        plt.title(name)
    else:
        plt.contourf(angles, omegas, policy_array.T, cmap='cividis')
        # proxy = [plt.Rectangle((0, 0), 1, 1, fc='gold'),
        #          plt.Rectangle((0, 0), 1, 1, fc='mediumblue')]
        # plt.legend(proxy, ["push right", "push left"])
        name = f'Greedy Policy of the DQN, Cart Velocity = {cart_velocity}'
        plt.title(name)
    plt.xlabel("angle")
    plt.legend()
    plt.ylabel("angular velocity")
    # plt.savefig(f'figures/{name}.png', dpi=600)
    plt.show()

