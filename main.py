# Hidden markov model
# Ali Behrouzi
# Hw 5

import numpy as np
from matplotlib import pyplot as plt

# from scipy.stats import norm

# Define the transition matrix
A = np.array([[1/4, 2/4, 1/4],
                [1/5, 1/5, 3/5],
                [1/6, 3/6, 2/6]])

# A = np.array([[1 / 3, 1 / 3, 1 / 3],
#               [1 / 3, 1 / 3, 1 / 3],
#               [1 / 3, 1 / 3, 1 / 3]])

# Define the emission matrix
B = np.array([[0.5, 0.5],
              [0.9, 0.1],
              [0.1, 0.9]])

# Define the initial state distribution
pi = np.array([1 / 3, 1 / 3, 1 / 3])

# Define the number of tosses
n = 100
# empty states sequense array
states = []
# empty observations sequense array
observations = []
# empty time array
times = []

# choose first state
state = np.random.choice([1, 2, 3], p=pi)
# append first state to states array
states.append(state)

# choose first observation
observation = np.random.choice(['H', 'T'], p=B[state - 1])

# append first observation to observations array
observations.append(observation)

time = 1
# append first time to times array
times.append(time)


# loop over the number of tosses
for i in range(n - 1):
    time = time + 1
    times.append(time)
    # choose next state
    state = np.random.choice([1, 2, 3], p=A[state - 1])

    # append next state to states array
    states.append(state)

    # choose next observation
    observation = np.random.choice(['H', 'T'], p=B[state - 1])

    # append next observation to observations array
    observations.append(observation)

    # print('time', time, ' State', state, ' : ', observation)

# plot the states in time
# numbers in the x and y axis not float
plt.figure(figsize=(10, 5))
plt.xticks(np.arange(0, n + 1, 1))
plt.yticks(np.arange(0, 4, 1))
plt.plot(times, states, 'o-')
plt.xlabel('time')
plt.ylabel('states')
plt.title('States in time')
plt.show()

# viterbi algorithm
# define the number of states
N = 3
# define the number of observations
M = 2
# define the number of time steps
T = times[-1]

# define the observations in numbers
observations_num = []
for i in range(len(observations)):
    if observations[i] == 'H':
        observations_num.append(0)
    else:
        observations_num.append(1)

# define the observations in numbers
observations_num = np.array(observations_num)

# define the delta matrix
delta = np.zeros((N, T))
# define the psi matrix
psi = np.zeros((N, T))

# initialize the delta matrix
delta[:, 0] = pi * B[:, observations_num[0]]

# initialize the psi matrix
psi[:, 0] = 0

# loop over the time steps
for t in range(1, T):
    # loop over the states
    for j in range(N):
        # calculate the delta matrix
        delta[j, t] = np.max(delta[:, t - 1] * A[:, j]) * B[j, observations_num[t]]
        # calculate the psi matrix
        psi[j, t] = np.argmax(delta[:, t - 1] * A[:, j]) + 1

# define the most probable states q*
most_probable_states = np.zeros(T)
# define the last state qT*
most_probable_states[-1] = np.argmax(delta[:, -1]) + 1

# loop over the time steps
for t in range(T - 2, -1, -1):
    # calculate the most probable states
    most_probable_states[t] = psi[int(most_probable_states[t + 1] - 1), t + 1]

# plot the most probable states in time
# numbers in the x and y axis not float
plt.figure(figsize=(10, 5))
plt.xticks(np.arange(0, n + 1, 1))
plt.yticks(np.arange(0, 4, 1))
plt.plot(times, most_probable_states, 'o-')
plt.xlabel('time')
plt.ylabel('states')
plt.title('Most probable states in time')
plt.show()

# rate of similarity between the most probable states and the real states
rate = 0
for i in range(len(states)):
    if states[i] == most_probable_states[i]:
        rate = rate + 1

rate = rate * 100 / len(states)
print('similarity rate between q and the real states is : ', rate)

# plot states and most probable states in time in one plot with different colors
# numbers in the x and y axis not float
plt.figure(figsize=(10, 5))
plt.xticks(np.arange(0, n + 1, 1))
plt.yticks(np.arange(0, 4, 1))
plt.plot(times, states, 'o-', color='red', label='real states')
plt.plot(times, most_probable_states, 'o-', color='blue', label='viterbi states')
plt.xlabel('time')
plt.ylabel('states')
plt.title('Real States and most probable states in time')
plt.legend()
plt.show()

# # estimate parameter of the model with Baum-Welch algorithm
# # define the number of iterations
# iterations = 100
#
# # define the initial transition matrix
#
# A_est = np.array([[1/4, 2/4, 1/4],
#                 [2/5, 1/5, 2/5],
#                 [2/6, 3/6, 1/6]])
#
#
# # define the initial emission matrix
# B_est = np.array([[0.5, 0.5],
#                   [0.5, 0.5],
#                   [0.5, 0.5]])
#
# # define the initial state distribution
# pi_est = np.array([1 / 3, 1 / 3, 1 / 3])
#
# # loop over the number of iterations
# for i in range(iterations):
#     # forward algorithm
#     # define the alpha matrix
#     alpha = np.zeros((N, T))
#     # initialize the alpha matrix
#     alpha[:, 0] = pi_est * B_est[:, observations_num[0]]
#     # loop over the time steps
#     for t in range(1, T):
#         # loop over the states
#         for j in range(N):
#             # calculate the alpha matrix
#             alpha[j, t] = np.sum(alpha[:, t - 1] * A_est[:, j]) * B_est[j, observations_num[t]]
#
#     # backward algorithm
#     # define the beta matrix
#     beta = np.zeros((N, T))
#     # initialize the beta matrix
#     beta[:, -1] = 1
#     # loop over the time steps
#     for t in range(T - 2, -1, -1):
#         # loop over the states
#         for j in range(N):
#             # calculate the beta matrix
#             beta[j, t] = np.sum(beta[:, t + 1] * A_est[j, :] * B_est[:, observations_num[t + 1]])
#
#     # define the gamma matrix
#     gamma = np.zeros((N, T))
#     # loop over the time steps
#     for t in range(T):
#         # calculate the gamma matrix
#         gamma[:, t] = alpha[:, t] * beta[:, t] / np.sum(alpha[:, t] * beta[:, t])
#
#     # define the ksi matrix
#     ksi = np.zeros((N, N, T - 1))
#     # loop over the time steps
#     for t in range(T - 1):
#         # calculate the ksi matrix
#         ksi[:, :, t] = (alpha[:, t].reshape(-1, 1) * A_est * B_est[:, observations_num[t + 1]].reshape(1, -1) * beta[:,
#                                                                                                                 t + 1].reshape(
#             1, -1)) / np.sum(alpha[:, t] * beta[:, t])
#
#     # re-estimate the transition matrix
#     A_est = np.sum(ksi, 2) / np.sum(gamma[:, :-1], 1).reshape(-1, 1)
#
#     # re-estimate the emission matrix
#     B_est = np.copy(B)
#     # loop over the states
#     for i in range(N):
#         # re-estimate the emission matrix
#         B_est[i,0] = np.sum(gamma[i, observations_num == 0]) / np.sum(gamma[i, :])
#         B_est[i,1] = np.sum(gamma[i, observations_num == 1]) / np.sum(gamma[i, :])
#
#     # re-estimate the state distribution
#     pi_est = gamma[:, 0] / np.sum(gamma[:, 0])
#
#
#
# # print the estimated parameters
# print('estimated transition matrix is : ', A_est)
# print('estimated emission matrix is : ', B_est)
# print('estimated state distribution is : ', pi_est)

