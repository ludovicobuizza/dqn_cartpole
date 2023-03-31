from trainer import Trainer
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    param_dict = {
        'policy_net_neurons': [4, 64, 64, 2],
        'target_net_neurons': [4, 64, 64, 2],
        'epsilon': 0.9,
        'epsilon_decay': 1E-4,
        'batch_size': 32,
        'num_episodes': 500,
        'update_target_every': 10,
        'ddqn': True,
        'learning_rate': 1E-3
    }
    trainer = Trainer(param_dict=param_dict, env=env)
    trainer.train()
    trainer.plot_durations()
    trainer.save_frames_as_gif(trainer.frames[-1], filename='cartpole.gif')
    print('Done')

