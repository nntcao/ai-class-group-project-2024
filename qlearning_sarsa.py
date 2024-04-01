"""
Explanation for Q algo:

1. Iterate over each action in the current policy
2. for each action, calculate the next state:
    and assigns a reward based on the transition function and certain conditions. 
    If the current state is '3' and the action is 'd' (dropoff) or if the current state is '4' and the action is 'p' (pickup), 
    the reward is set to a predefined pickup/dropoff reward. 
    Otherwise, it assigns a negative penalty for moving.



Max Q-value of Next State: It calculates the maximum Q-value for the next state over all possible actions that can be taken from that state. This is a key step in the Q-learning algorithm, as it estimates the maximum expected future reward from the next state.

Updating Q-value:  It updates the Q-value for the current state-action pair using the Q-learning update rule: Q(s,a) <- Q(s,a) + alpha[r, + gamma * max_a'Q(s', a') - Q(s,a)]
    Q(s,a) = q-val of current state-action pair
    r = reward obtained from taking action a in state  
    alpha = learning rate
    gamma = discount factor
    s' is then next state 
    max_a'Q(s', a') is the max q value for the next state 

Then we update the current state to the next state, preparing for the next iteration 'current_state = next_state'

"""

def Q_learning(policy, transition, Q_table, gamma, alpha, move_penalty, pickup_dropoff_reward, actions, initial_state):
    current_state = initial_state
    for action in policy:
        next_state = transition(current_state, action)
        if current_state == "3'" and action == 'd':
            reward = pickup_dropoff_reward
        elif current_state == '4' and action == 'p':
            reward = pickup_dropoff_reward
        else:
            reward = -move_penalty
        max_q_next = max(Q_table[(next_state, a)] for a in actions if Q_table(next_state, a) is not None)
        Q_table[(current_state, action)] += alpha * (reward + gamma + max_q_next - Q_table[(current_state, action)])
        current_state = next_state

def SARSA(policy, transition, Q_table, gamma, alpha, move_penalty, pickup_dropoff_reward, actions, initial_state):
    current_state = initial_state
    current_action = None
    for action in policy:
        next_state = transition(current_state, action)
        if current_state == "3'" and action == 'd':
            reward = pickup_dropoff_reward
        elif current_state == '4' and action == 'p':
            reward = pickup_dropoff_reward
        else:
            reward = -move_penalty
        if current_action is not None:
            next_action = current_action
            next_q = Q_table[(next_state, next_action)]
        else:
            next_action = None
            next_q = 0  # Terminal state
        Q_table[(current_state, action)] += alpha * (reward + gamma * next_q - Q_table[(current_state, action)])
        current_state = next_state
        current_action = next_action

def transition(current_state, action):
    
    # Assuming a grid world with coordinates (x, y)
    x, y = current_state
    
    # Update the state based on the action
    if action == 'u':  # Move up
        next_state = (x, y + 1)
    elif action == 'd':  # Move down
        next_state = (x, y - 1)
    elif action == 'l':  # Move left
        next_state = (x - 1, y)
    elif action == 'r':  # Move right
        next_state = (x + 1, y)
    else:
        raise ValueError("Invalid action")
    
    # Check if the next state is valid (within the grid boundaries)
    # You might want to handle edge cases differently (e.g., if the agent reaches a wall)
    # Here, let's assume an infinite grid where all states are valid
    return next_state# Define the Q-table


Q_table = {}  # Initialize as an empty dictionary

policy = ['u', 'd', 'l', 'r']  # Example: Up, Down, Left, Right
alpha = 0.3  
gamma = 0.5  
move_penalty = 0.1  # Example: 0.1
pickup_dropoff_reward = 1  # Example: 1
actions = ['u', 'd', 'l', 'r']  # Example: Up, Down, Left, Right
initial_state = (0,0) 
Q_learning(policy, transition, Q_table, gamma, move_penalty, pickup_dropoff_reward, actions, initial_state, alpha)
