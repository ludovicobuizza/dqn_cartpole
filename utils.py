import random
import torch
from dqn import DQN


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
        bellman_targets = (~dones).reshape(-1) * \
                          (target_dqn(next_states).gather(1,
                                                          best_action_index)).\
                              reshape(-1) + rewards.reshape(-1)
    else:
        bellman_targets = (~dones).reshape(-1) * (target_dqn(next_states)).max(
            1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets) ** 2).mean()

