# Learn2Slither
Learn2Slither

[https://8thlight.com/insights/qlearning-teaching-ai-to-play-snake](https://8thlight.com/insights/qlearning-teaching-ai-to-play-snake)

[https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c](https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c)

https://github.com/cmariot/Learn2Slither

### Résultats
| Model | couches |          vue              | Nb Game | durée | Best score | Moyenne |
| 100K2 |   4     | distance premier obstacle |  100K   |  15h  |    17      | |
| 100K3 |   4     | Distance de ts obstacles  |  100K   |  33h  |    15      | 2.5  |
| 100K4 |   1     |       même chose          |  100K   |  12h  |     18     | 2.34 |
essais avec supression dans model de la réduction impacte pénalité avec gamma 0.9