import numpy as np
import tqdm

from wordle.base_policy import Policy
from wordle.environment import WordleEnv
from wordle.expert_policy import ExpertPolicy


def evaluate(env: WordleEnv, policy: Policy, num_games: int) -> None:
    episode_rewards = []
    episode_steps = []
    failed_episodes = []
    for _ in tqdm.tqdm(range(num_games)):
        states, _ = env.reset()
        episode_reward = 0
        num_steps = 0
        terminal_state = False

        while not terminal_state:
            action = policy.predict(states=states)
            states, reward, terminal_state, _, _ = env.step(action)
            episode_reward += reward
            num_steps += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(num_steps)
        if episode_reward == 0:
            failed_episodes.append((states, env.hidden_word))

    print(f"Average reward {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}")
    print(f"Max reward {np.max(episode_rewards)}")
    print(f"Average steps {np.mean(episode_steps):.3f} +/- {np.std(episode_steps):.3f}")
    print(f"Failed episodes {len(failed_episodes)}")
    for states, hidden_word in failed_episodes:
        print(f"Hidden word {hidden_word}, states {states}")


if __name__ == "__main__":
    env = WordleEnv()
    env.reset(seed=0)
    policy = ExpertPolicy()
    evaluate(env, policy, num_games=100)
