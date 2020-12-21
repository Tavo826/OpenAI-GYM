'''
    PROBANDO EL GYM DE OPENAI

Entorno de aprendizaje reforzado,
importando alguno de los entornos (juegos)

state: estado en el que se encuentra (0 a 499)
reward: variable binaria que recompensa mientras si le logra un objetivo 
done: variable booleana para resetear el entorno
info: información para debuguear (el objeto del agente no tiene acceso a esta variable)
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
env.reset()

episodes = 250

alpha = 0.618
Q = np.zeros([env.observation_space.n, env.action_space.n])
tR = 0
rewards=([])

for episode in range(1,episodes):
    done = False
    recompensa, reward = 0,0
    state = env.reset()
    while done != True:
            action = np.argmax(Q[state]) #1
            state2, reward, done, info = env.step(action) #2
            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
            recompensa += reward
            state = state2

            rewards.append(reward)

    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,recompensa))
        

print(Q)
plt.plot(range(len(rewards)), rewards) #Curva de aprendizaje
plt.show()
env.close()