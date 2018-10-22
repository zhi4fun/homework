import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkldir', nargs='*')
    parser.add_argument('--legend', '-l', type=str, nargs='*')
    parser.add_argument('--maxstep', '-m', type=int, default=3e6)
    args = parser.parse_args()

    steps = []
    mean_epi_rewards = []
    best_mean_epi_rewards = []
    legend = []
    for pkldir in args.pkldir:
    	with open(pkldir, 'rb') as f:
    		data = pickle.load(f)
    		steps.append([i[0] for i in data])
    		mean_epi_rewards.append([i[1] for i in data])
    		best_mean_epi_rewards.append([i[2] for i in data])

    plt.figure()
    for i in range(len(steps)): 
        plt.plot(steps[i], mean_epi_rewards[i])
        plt.plot(steps[i], best_mean_epi_rewards[i])
        legend.append('mean 100-episode reward - ' + args.legend[i])
        legend.append('best mean 100-episode reward - ' + args.legend[i])
    plt.grid(True)
    plt.xlim(0, args.maxstep)
    plt.xlabel('Time steps')
    plt.ylabel('Mean_episode_rewards')
    plt.title('DQN implementation on the game Pong')
    plt.legend(legend, loc='lower right')
    plt.show()
    '''

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('pkldir')
	args = parser.parse_args()

	with open(args.pkldir, 'rb') as f:
		data = pickle.load(f)
		mean_epi_rewards = []
		best_mean_epi_rewards = []
		for i in range(len(data)):
			x = data[:i+1]
			mean_epi_rewards.append(np.mean(x[-100:]))
			best_mean_epi_rewards.append(max(mean_epi_rewards))
	plt.figure()
	plt.plot(mean_epi_rewards)
	plt.plot(best_mean_epi_rewards)
	plt.grid(True)
	plt.xlabel('Episode')
	plt.ylabel('Mean_episode_rewards')
	plt.title('DQN implementation on the game Pong')
	plt.legend(('mean 100-episode reward', 'best mean 100-episode reward'), loc='lower right')
	plt.show()'''


if __name__ == "__main__":
    main()