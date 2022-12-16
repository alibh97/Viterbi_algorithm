# Hidden markov model
# Ali Behrouzi
# Hw 5

import numpy as np
from matplotlib import pyplot as plt
# from scipy.stats import norm

# Define the transition matrix
A = np.array([[1/3, 1/3, 1/3],
                [1/3, 1/3, 1/3],
                [1/3, 1/3, 1/3]])

# Define the emission matrix
B = np.array([[0.5, 0.5],
                [0.75, 0.25],
                [0.25, 0.75]])

# Define the initial state distribution
pi = np.array([1/3, 1/3, 1/3])

# Define the number of tosses
n = 20
# empty states sequense array
states = []
# empty observations sequense array
observations = []
# empty time array
times = []

# choose first state
state = np.random.choice([1, 2,3], p=pi)
# append first state to states array
states.append(state)

# choose first observation
observation = np.random.choice(['H', 'T'], p=B[state-1])

# append first observation to observations array
observations.append(observation)

time=1
# append first time to times array
times.append(time)

print('time',time,' State',state,' : ', observation)

# loop over the number of tosses
for i in range(n-1):
    time=time+1
    times.append(time)
    # choose next state
    state = np.random.choice([1, 2,3], p=A[state-1])

    # append next state to states array
    states.append(state)

    # choose next observation
    observation= np.random.choice(['H', 'T'], p=B[state-1])

    # append next observation to observations array
    observations.append(observation)

    print('time',time,' State',state,' : ', observation)


# plot the states in time
# numbers in the x and y axis not float
plt.figure(figsize=(10, 5))
plt.xticks(np.arange(0, n+1, 1))
plt.yticks(np.arange(0, 4, 1))
plt.plot(times, states, 'o-')
plt.xlabel('time')
plt.ylabel('states')
plt.title('States in time')
plt.show()


