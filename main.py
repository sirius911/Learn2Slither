from game import SnakeGameAI
from Agent import Agent
from helper import plot
import argparse


def train(learning=True):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    # Charger un modèle existant si spécifié
    if not learning:
        agent.model.load('model_100_sessions.pth') 

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)

        if learning:
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            if learning:
                
                agent.train_long_memory()

                if agent.n_games in [1, 10, 100]:
                # Sauvegarder après 1, 10, 100 sessions
                    file_name = f'model_{agent.n_games}_sessions.pth'
                    agent.model.save(file_name)
                    print(f'Model saved: {file_name}')

                if score > record:
                    record = score
                    agent.model.save('model_best.pth')  # Sauvegarde du meilleur score
                    print('Best model saved!')

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


# if __name__ == '__main__':
#     train()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate the Snake AI.')
    parser.add_argument('--load', type=str, help='Path to the model to load.')
    parser.add_argument('--no-learn', action='store_true', help='Disable learning mode.')
    args = parser.parse_args()

    if args.load:
        agent = Agent()
        agent.model.load(args.load)
        train(learning=not args.no_learn)
    else:
        train()
