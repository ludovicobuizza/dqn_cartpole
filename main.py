from trainer import Trainer
import gymnasium as gym
from utils import plot_average_durations

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    param_dict = {
        'policy_net_neurons': [4, 128, 128, 128, 2],
        'target_net_neurons': [4, 128, 128, 128, 2],
        'epsilon': 1,
        'epsilon_decay': 1E-3,
        'batch_size': 64,
        'num_episodes': 500,
        'update_target_every': 5,
        'ddqn': False,
        'learning_rate': 1E-3
    }
    results = []
    for i in range(10):
        print('Training run', i)
        trainer = Trainer(param_dict=param_dict, env=env)
        trainer.train()
        trainer.save_frames_as_gif(trainer.frames[-1], filename=f'cartpole{i}.gif')
        results.append(trainer.episode_durations)
        print('Last duration:', trainer.episode_durations[-1])
    plot_average_durations(results)
    print('Done')

