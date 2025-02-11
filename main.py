from game import SnakeGameAI
from Agent import Agent
from helper import plot, print_tensor
from directions import Direction
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


def ending(agent, learning=False, save="last_model.pth"):
    if learning:
        agent.model.save(file_name=save)
        plt.savefig("training_results.png")
    print("fin du programme")


def play(agent=None, learning=True,
         verbose=False, graphique=True,
         step=False, save="last_model.pth"):
    plot_scores = []
    plot_mean_scores = []
    epsilon_values = []
    total_score = 0
    record = 0
    game = SnakeGameAI(verbose=verbose,
                       graphique=graphique,
                       back_function=lambda: ending(agent, learning))
    if agent is None:
        agent = Agent()
    max_duration = 0
    current_duration = 0
    try:
        for session in tqdm(range(SESSIONS), desc='Training' if learning else 'gaming'):
            done = False
            while not done:
                state_old = game.get_state()
                if step:
                    game.print_snake_vision()
                    print_tensor(state_old)
                move: int = agent.get_action(state=state_old, learning=learning)
                absolute_move = Direction.directions()[move]
                direction_move = Direction.relative_direction(game.direction, absolute_move)
                if step:
                    print(f"direction : {game.direction.name}, relative_move = {direction_move.name}")
                    game.wait()

                current_duration += 1
                reward, done, score = game.play_step(absolute_move)

                if learning:
                    state_new = game.get_state()
                    agent.train_short_memory(state_old, move, reward, state_new, done)
                    agent.remember(state_old, move, reward, state_new, done)

            # Done = True
            total_score += score
            if score > record:
                record = score
                game.save_map()
                game.best_score = record
                max_duration = max(max_duration, current_duration)
            current_duration = 0
            game.reset()
            agent.n_games += 1
            game.nb_game = agent.n_games

            # prop_random, prop_neurone = calculate_proportions(agent.n_games)
            # plot_prop_random.append(prop_random)
            # plot_prop_neurone.append(prop_neurone)

            if learning:
                epsilon_values.append(round(agent.epsilon * 100, 2))
                agent.train_long_memory()

                if agent.n_games in [1, 10, 100, 1000, 10000, 100000]:
                    agent.model.save(f'{save}_{agent.n_games}_sessions.pth')

            if verbose:
                print(f"Game #{agent.n_games}, Best Score: {record}, Max duration : {max_duration}")
            plot_scores.append(score)
            plot_mean_scores.append(total_score / agent.n_games)

            if graphique and learning and not step:
                plot(plot_scores, plot_mean_scores,
                     epsilon_values, learning)
    except KeyboardInterrupt:
        print("\n Interrupted by Ctrl-C")
        if learning:
            agent.model.save(f'{save}_{agent.n_games}_sessions.pth')
    finally:
        if agent.n_games > 0:
            mean_score = total_score / agent.n_games
        else:
            mean_score = 0.0
        print(f"number of games : {agent.n_games}, Best score = {record}, \
              Max duration : {max_duration} mean score = {mean_score:0.2f} \
               Nb boucle infinie : {game.nb_infini}")
        if not graphique or not learning:
            plot(plot_scores, plot_mean_scores,
                 epsilon_values, learning)

        ending(agent=agent, learning=learning, save=save)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or evaluate the Snake AI.')
    parser.add_argument('--load', type=str, help='Path to the model to load.')
    parser.add_argument('--save', type=str, help="Filename of model to save.")
    parser.add_argument('--sessions', type=int, default=100, help="Nombre de sessions de jeux")
    parser.add_argument('--no-learn', action='store_true', help='Disable learning mode.')
    parser.add_argument("--verbose", action='store_true', help="Mode verbeux et graphique")
    parser.add_argument("--no-graphic", action='store_true', help="mode non graphique")
    parser.add_argument("--step", action='store_true', help="show step by step the learning")
    args = parser.parse_args()
    SESSIONS = int(args.sessions)
    if args.load:
        agent = Agent()
        agent.model.load(args.load)
        play(agent=agent, learning=not args.no_learn, verbose=args.verbose,
             graphique=not args.no_graphic, step=args.step, save=args.save)
    else:
        play(agent=None, learning=not args.no_learn, verbose=args.verbose,
             graphique=not args.no_graphic, step=args.step, save=args.save)
    input("taper ENTER pour fermer")
