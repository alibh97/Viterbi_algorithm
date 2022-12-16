# Hidden markov model
# Ali Behrouzi
# Hw 5

import numpy as np
from matplotlib import pyplot as plt


def observe():
    if state == 'S1':
        o = np.random.choice(['H', 'T'], p=B[0])
    elif state == 'S2':
        o = np.random.choice(['H', 'T'], p=B[1])
    else:
        o = np.random.choice(['H', 'T'], p=B[2])
    return o


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

# choose first state
state = np.random.choice(['S1', 'S2', 'S3'], p=pi)
# append first state to states array
states.append(state)

# choose first observation
observation=observe()

# append first observation to observations array
observations.append(observation)

print('time',1,' State',state,' : ', observation)

# loop over the number of tosses
for i in range(n-1):
    # choose next state
    if state == 'S1':
        state = np.random.choice(['S1', 'S2', 'S3'], p=A[0])
    elif state == 'S2':
        state = np.random.choice(['S1', 'S2', 'S3'], p=A[1])
    else:
        state = np.random.choice(['S1', 'S2', 'S3'], p=A[2])

    # append next state to states array
    states.append(state)

    # choose next observation
    observation=observe()

    # append next observation to observations array
    observations.append(observation)

    print('time',i+2,' State',state,' : ', observation)



# # Define the observations
# observations = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
#
# # Define the number of states
# num_states = A.shape[0]
# #
# # Define the number of observations
# num_observations = len(observations)
#
# # Define the number of iterations
# num_iterations = 10
#
# # Define the number of frames
# num_frames = num_iterations + 1
#
# # Define the figure
# fig = plt.figure()
#
# # Define the axes
# ax = fig.add_subplot(111)
#
# # Define the title
# ax.set_title('Hidden Markov Model')
#
# # Define the x-axis
# ax.set_xlabel('Observations')
#
# # Define the y-axis
# ax.set_ylabel('States')
#
# # Define the x-axis limits
# ax.set_xlim(0, num_observations)
#
# # Define the y-axis limits
# ax.set_ylim(0, num_states)
#
# # Define the x-axis ticks
# ax.set_xticks(range(num_observations))
#
# # Define the y-axis ticks
# ax.set_yticks(range(num_states))
#
# # Define the x-axis tick labels
# ax.set_xticklabels(observations)
#
# # Define the y-axis tick labels
# ax.set_yticklabels(range(num_states))
#
#
# fig.show()

