from game import SnakeGameAI
from Agent import Agent
from helper import plot, get_state, calculate_proportions, calc_epsilon
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

SESSIONS=2000

def ending(agent):
    agent.model.save('last_model.pth')
    plt.savefig("training_results.png")
    print("fin du programme")

def play(learning=True, verbose=False, graphique=True, step=False):
    plot_scores = []
    plot_mean_scores = []
    epsilon_values = []
    plot_prop_random = []
    plot_prop_neurone = []
    total_score = 0
    total_reward = 0
    record = 0
    game = SnakeGameAI(verbose=verbose,
                       graphique=graphique,
                       back_function=lambda: ending(agent))
    agent = Agent()
    max_duration = 0
    current_duration = 0

    for session in tqdm(range(SESSIONS), desc='Training' if learning else 'gaming'):
        done = False
        while not done:
            state_old = get_state(game)
            final_move = agent.get_action(state=state_old, game=game, learning=learning)
            current_duration += 1
            reward, done, score = game.play_step(final_move, step)
            total_score += score

            if learning:
                state_new = get_state(game)
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)

            total_reward += reward

            if done:

                max_duration = max(max_duration, current_duration)
                current_duration = 0
                game.reset()
                agent.n_games += 1

                epsilon_values.append(agent.epsilon)
                
                prop_random, prop_neurone = calculate_proportions(agent.n_games)
                plot_prop_random.append(prop_random)
                plot_prop_neurone.append(prop_neurone)
                if learning:
                    agent.train_long_memory()

                mean_reward = total_reward / agent.n_games

                if agent.n_games in [1, 10, 100]:
                    agent.model.save(f'model_{agent.n_games}_sessions.pth')

                if score > record:
                    record = score
                    agent.model.save('model_best.pth')

                if verbose:
                    print(f"Game {agent.n_games}, Best Score: {record}, Mean Score: {total_score / agent.n_games:0.2f}, Mean Reward: {mean_reward:.2f}, "
                        f"Memory Size: {len(agent.memory)}, Exploration: {agent.exploration_count}, Exploitation: {agent.exploitation_count}")
                plot_scores.append(score)
                plot_mean_scores.append(total_score / agent.n_games)

                if graphique:                                        
                    plot(plot_scores, plot_mean_scores,
                        plot_prop_random, plot_prop_neurone,
                        epsilon_values, title=('Training' if learning else 'gaming'))
    if not verbose:
        print(f"Game {agent.n_games}, Best Score: {record}, Mean Score: {total_score / agent.n_games:0.2f}, Mean Reward: {mean_reward:.2f}, ")
    if not graphique:
        plot(plot_scores, plot_mean_scores,
            plot_prop_random, plot_prop_neurone,
            epsilon_values, title=('Training' if learning else 'gaming'))
    ending(agent=agent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate the Snake AI.')
    parser.add_argument('--load', type=str, help='Path to the model to load.')
    parser.add_argument('--session', type=int, default=100, help="Nombre de session de jeux")
    parser.add_argument('--no-learn', action='store_true', help='Disable learning mode.')
    parser.add_argument("--verbose", action='store_true', help="Mode verbeux et graphique")
    parser.add_argument("--no-graphic", action='store_true', help="mode non graphique")
    parser.add_argument("--step", action='store_true', help="show step by step the learning")
    args = parser.parse_args()
    SESSIONS = int(args.session)
    if args.load:
        agent = Agent()
        agent.model.load(args.load)
        play(learning=not args.no_learn, verbose=args.verbose,
             graphique=not args.no_graphic, step=args.step)
    else:
        play(learning=not args.no_learn, verbose=args.verbose,
             graphique=not args.no_graphic, step=args.step)
    input("taper ENTER pour fermer")