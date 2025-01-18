from game import SnakeGameAI
from Agent import Agent
from helper import plot, get_state, calculate_proportions, calc_epsilon
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

SESSIONS=2000

def ending(agent, learning=False):
    if learning:
        agent.model.save('last_model.pth')
        plt.savefig("training_results.png")
    print("fin du programme")

def play(agent=None, learning=True, verbose=False, graphique=True, step=False):
    plot_scores = []
    plot_mean_scores = []
    epsilon_values = []
    plot_prop_random = []
    plot_prop_neurone = []
    losses = []
    total_score = 0
    total_reward = 0
    record = 0
    game = SnakeGameAI(verbose=verbose,
                       graphique=graphique,
                       back_function=lambda: ending(agent, learning))
    if agent is None:
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
                loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
                losses.append(loss)
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
                if score > record:
                    record = score
                if learning:
                    agent.train_long_memory()

                    if agent.n_games in [1, 10, 100]:
                        agent.model.save(f'model_{agent.n_games}_sessions.pth')

                if verbose:
                    print(f"Game {agent.n_games}, Best Score: {record}, Mean Score: {total_score / agent.n_games:0.2f}, "
                        f"Memory Size: {len(agent.memory)}, Exploration: {agent.exploration_count}, Exploitation: {agent.exploitation_count}")
                plot_scores.append(score)
                plot_mean_scores.append(total_score / agent.n_games)

                if graphique and not learning:                                        
                    plot(plot_scores, plot_mean_scores,
                        plot_prop_random, plot_prop_neurone,
                        epsilon_values, losses, learning)
    if not verbose:
        print(f"Game {agent.n_games}, Best Score: {record}, Mean Score: {total_score / agent.n_games:0.2f}, ")
    if not graphique:
        plot(plot_scores, plot_mean_scores,
            plot_prop_random, plot_prop_neurone,
            epsilon_values, losses, learning)
    if learning:
        mean_loss = sum(losses) / len(losses) if losses else 0
        if len(losses) >= 100:
            moving_avg_loss = sum(losses[-100:]) / 100
        else:
            moving_avg_loss = sum(losses) / len(losses) if losses else 0

        print(f"Training Summary:")
        print(f"  - Mean Loss: {mean_loss:.4f}")
        print(f"  - Last Loss: {losses[-1]:.4f}")
        print(f"  - Moving Avg Loss (Last 100): {moving_avg_loss:.4f}")
        print(f"  - Min Loss: {min(losses):.4f}, Max Loss: {max(losses):.4f}")

    ending(agent=agent, learning=learning)

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
        play(agent=agent, learning=not args.no_learn, verbose=args.verbose,
             graphique=not args.no_graphic, step=args.step)
    else:
        play(agent=None, learning=not args.no_learn, verbose=args.verbose,
             graphique=not args.no_graphic, step=args.step)
    input("taper ENTER pour fermer")