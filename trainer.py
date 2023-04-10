import torch
from matplotlib import animation
import numpy as np
from replay_buffer import ReplayBuffer
from utils import epsilon_greedy, update_target, loss
from dqn import DQN
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """Trainer class for DQN training."""
    def __init__(self, env, param_dict):
        """Initialise trainer class.

        Args: env: environment to train on
        param_dict: dictionary with parameters for training.
            These are: policy_net neurons per layer (
            need output 2), target_net neurons per layer (need output 2),
            epsilon, epsilon_decay, batch_size, num_episodes, num_steps,
            update_target_every, ddqn bool, learning_rate
        """
        self.policy_net = DQN(param_dict['policy_net_neurons'])
        self.target_net = DQN(param_dict['target_net_neurons'])
        update_target(self.target_net, self.policy_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=param_dict['learning_rate'])
        self.env = env
        self.replay_buffer = ReplayBuffer(10000)
        self.loss = loss
        self.epsilon = param_dict['epsilon']
        self.epsilon_decay = param_dict['epsilon_decay']
        self.batch_size = param_dict['batch_size']
        self.num_episodes = param_dict['num_episodes']
        self.update_target_every = param_dict['update_target_every']
        self.ddqn = param_dict['ddqn']
        self.episode_durations = []
        self.frames = []

    def train(self):
        pbar = tqdm(range(self.num_episodes))
        for ep in pbar:
            ep_frames = []
            obs, info = self.env.reset()
            state = torch.Tensor(obs)
            truncated, terminated = (False, False)
            t = 0
            self.epsilon = max(self.epsilon_decay * self.epsilon, 0.05)
            while not (truncated or terminated):
                ep_frames.append(self.env.render())
                state, truncated, terminated = self._step(state)

                if self.batch_size < len(self.replay_buffer.buffer):
                    self._optimize()
                if truncated or terminated:
                    self.episode_durations.append(t + 1)
                t += 1
            if ep % self.update_target_every == 0:
                update_target(self.target_net, self.policy_net)
            pbar.set_description(f"Episode {ep} - Average duration: {int(np.mean(self.episode_durations))}")
            if ep == 0 or ep == self.num_episodes - 1:
                self.frames.append(ep_frames)

    def plot_durations(self, filename: str = 'durations.png'):
        plt.figure()
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Episode Duration vs Training Time')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 25 episode averages and plot them too
        if len(durations_t) >= 25:
            means = durations_t.unfold(0, 25, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(24), means))
            plt.plot(means.numpy())
        plt.savefig(filename)


    def _optimize(self):
        transitions = self.replay_buffer.sample(batch_size=self.batch_size)
        state_batch, action_batch, nextstate_batch, reward_batch, dones = \
            (torch.stack(x) for x in zip(*transitions))
        mse_loss = loss(self.policy_net, self.target_net, state_batch,
                        action_batch, reward_batch, nextstate_batch, dones,
                        ddqn=self.ddqn)
        # Optimize the model
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

    def _step(self, state):
        action = epsilon_greedy(self.epsilon, self.target_net, state)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = torch.tensor([reward])
        action = torch.tensor([action])
        next_state = torch.tensor(obs).reshape(-1).float()
        done = terminated or truncated
        self.replay_buffer.push([state, action, next_state, reward,
                                 torch.tensor([done])])
        state = next_state
        return state, truncated, terminated

    @staticmethod
    def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
        # Mess with this to change frame size
        plt.figure(
            figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
            dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                       interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)
