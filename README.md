# Learn2Slither
Learn2Slither

[https://8thlight.com/insights/qlearning-teaching-ai-to-play-snake](https://8thlight.com/insights/qlearning-teaching-ai-to-play-snake)

[https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c](https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c)


### Résultats
| Model | couches  |          vue               | Nb entrainement | Nb game test | durée | Best score | Moyenne | Loop |
| best  | 16-256-4 | 4 x dist dangers + Green A |       15000     |      10000   |  753  |     63     |  18.61  | 2.9% |
| loop  | 16-256-4 | 4 x dist dangers + Green A |       15000     |      10000   |  640  |     54     |  17.14  | 3.3% | avec détection boucles infinies

infinity avec reward -300 < error reward sur 10000 parties
number of games : 10000, Best score = 55,               Max duration : 604 mean score = 11.38                Nb boucle infinie : 57,17%

infinity avec reward == error reward sur 10000 parties
number of games : 10000, Best score = 60,               Max duration : 529 mean score = 15.81                Nb boucle infinie : 39.72%

1er Essais avec reload sur 10000 entrainements meme infinity reward qu'au dessus:
number of games : 10000, Best score = 57,               Max duration : 584 mean score = 20.72                Nb boucle infinie : 28,81%

2eme Essais avec reload sur 10000 entrainements meme infinity reward qu'au dessus:
number of games : 10000, Best score = 51,               Max duration : 467 mean score = 16.53                Nb boucle infinie : 3663

Essais sur 300 000 entrainements sans reload::
number of games : 10000, Best score = 57,               Max duration : 534 mean score = 16.43                Nb boucle infinie : 33,47%