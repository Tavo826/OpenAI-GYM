'''
    PROBANDO EL GYM DE OPENAI

Entorno de aprendizaje reforzado,
importando alguno de los entornos (juegos)

observation: objeto que representa la 
    observación del entorno (cámara, ángulos y velocidades o estado del tablero)
reward: variable binaria que recompensa mientras si le logra un objetivo 
done: variable booleana para resetear el entorno
info: información para debuguear (el objeto del agente no tiene acceso a esta variable)
'''

import gym
import numpy as np
from gym import spaces

env = gym.make('CartPole-v1')
env.reset()

print(env.action_space)
print(env.observation_space.high) #Valores máximos del vector de observación
print(env.observation_space.low) #Valores mínimos del vector de observación

space = spaces.Discrete(8) #Generando un espacio con una distribución de probabilidad sobre 8 números discretos
print(space.sample()) #Probabilidad en ese espacio discreto

action_vector = np.array([])
episodes = 20
iters = 100

# Para determinar el número de acciones sin correr el entorno
#action_vector = np.array([env.action_space.sample() for _ in range(iters)])

# for i_episode in range(episodes):
#     observation = env.reset()
#     for t in range(iters):
#         env.render()
#         #print(observation) # Posición_Carro Velocidad_Carro Ángulo_Palo Velocidad_Palo_En_La_Punta
#         action = env.action_space.sample() #Es binaria
#         action_vector = np.append(action_vector, action)
#         observation, reward, done, info = env.step(action) # take a random action
#         if done:
#             #La idea es optimizar el timestep
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

#print(np.sum(action_vector) / iters) #Acciones

#--------------RESOLVIENDO A PEDAL 1-------------

# tR = 0

# for i_episode in range(episodes):
#     recompensa = 0
#     observation = env.reset()
#     for t in range(iters):
#         env.render()
#         #print(observation) # Posición_Carro Velocidad_Carro Ángulo_Palo Velocidad_Palo_En_La_Punta
#         #action = env.action_space.sample() #Es binaria
#         #action_vector = np.append(action_vector, action)
#         observation, reward, done, info = env.step(int(observation[2] > 0.0))
#         recompensa += reward
#         if done:
#             #La idea es optimizar el timestep
#             print("Episode finished after {} timesteps: {}".format(t+1, recompensa))
#             tR += recompensa
#             break
# print('Average reward: ', tR / episodes)
# env.close()

#Si la acción es aleatoria, la recompensa media es alrededor de 20
#Con esta solución esta media se duplica

#-----------------------------------------------

#--------------RESOLVIENDO A PEDAL 2 (Error con el done)-------------

tR = 0

for i_episode in range(episodes):
    recompensa = 0
    observation = env.reset()
    for t in range(iters):
        env.render()
        #print(observation) # Posición_Carro Velocidad_Carro Ángulo_Palo Velocidad_Palo_En_La_Punta
        action = int(observation[2] > 0.0 and observation[3] > 0.0)
        #action_vector = np.append(action_vector, action)
        observation, reward, done, info = env.step(action)
        recompensa += reward
        if done:
            #La idea es optimizar el timestep
            print("Episode finished after {} timesteps: {}".format(t+1, recompensa))
            tR += recompensa
            break
print('Average reward: ', tR / episodes)
env.close()

#Si la acción es aleatoria, la recompensa media es alrededor de 20
#Con esta solución esta media es alrededor de 160

#-----------------------------------------------