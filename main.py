from enum import Enum
import numpy as np
import random
from copy import deepcopy
import pandas as pd
from collections import defaultdict
import csv

class Position:
    """A class used to store position data
    
    Attributes:
        x (int): The x value of the position vector
        y (int): The y value of the position vector
    """

    def __init__(self, x: int = 0, y: int = 0) -> None:
        """Initialize position data"""
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """Convert Position to string of format '(x, y)'"""
        return "(" + str(self.x) + ", " + str(self.y) + ")"


class SpaceType(Enum):
    EMPTY = "E"
    DROP_OFF = "D"
    PICK_UP = "P"


class Space:
    """A class used to store Space data, like a Space on a tabletop board game
    
    Attributes:
        type (SpaceType): represents the type of space encoding e.g. Empty, Pick Up, Drop Off
        num_blocks (int): the number of blocks sitting on the space
        max_blocks (int): the maximum number of blocks that can sit on the space
        reward (int): the reward the space gives
    """

    def __init__(self, type: SpaceType = SpaceType.EMPTY, num_blocks: int = 0, max_blocks: int = 5) -> None:
        """Initialize Space class"""
        if num_blocks > max_blocks:
            raise ValueError("max blocks exceeded on space")
        if num_blocks < 0:
            raise ValueError("num blocks cannot be less than 0 on space")
        if max_blocks < 0:
            raise ValueError("max blocks cannot be less than 0 on space")
        self.type = type
        self.num_blocks = num_blocks
        self.max_blocks = max_blocks
        self.reward = 0
        self._init_reward()

    def _init_reward(self):
        """Initialize reward based on type (SpaceType)"""
        if self.type == SpaceType.EMPTY:
            self.reward = -1
        elif self.type == SpaceType.DROP_OFF:
            self.reward = 13
        elif self.type == SpaceType.PICK_UP:
            self.reward = 13

    
    def __str__(self) -> str:
        """Convert Space to string of format '(SpaceType, num_blocks, max_blocks)'"""
        return "(" + str(self.type) + "," + str(self.num_blocks) + "," + str(self.max_blocks) + ")"

class Environment:
    """Environment class representing a 2D board of Spaces

    Attributes:
        n (int): the number of rows in the 2D environment
        m (int): the number of cols in the 2D environment
        pd_world (list[list[Space]]): the board of Spaces
    
    """

    def __init__(self, n: int = 5, m: int = 5) -> None:
        """Initialize the Environment world"""
        self.n = n
        self.m = m
        self.pd_world = [[Space(max_blocks=0)
                          for _ in range(m)] for _ in range(n)]

    def set(self, pos: Position, space: Space) -> None:
        """Set pos to be space in the environment"""
        if pos.x < 0 or self.n <= pos.x or pos.y < 0 or self.m <= pos.y:
            raise ValueError("cannot set space as position is out of bounds")
        self.pd_world[pos.x][pos.y] = space

    def at(self, pos: Position) -> Space:
        """Return the space at pos in the environment"""
        if pos.x < 0 or self.n <= pos.x or pos.y < 0 or self.m <= pos.y:
            raise ValueError("cannot find space as position is out of bounds")
        return self.pd_world[pos.x][pos.y]

    def within_bounds(self, pos: Position) -> None:
        """Checks if a Position pos is within the bounds of the environment"""
        return -1 < pos.x and pos.x < self.n and \
            -1 < pos.y and pos.y < self.m

    def __str__(self) -> str:
        """Generates a string representing the environment"""
        res = ""
        for i in range(self.m):
            for j in range(self.n):
                res += self.pd_world[i][j].type.value + " "
            res += "\n"
        return res


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5
    IDLE = 6



class ActorType(Enum):
    RED = "R"
    BLUE = "B"
    BLACK = "K"


class Actor:
    """Represents the Actor that can exist in an Environment

    Attributes:
        type (ActorType): represents the color of the Actor
        position (Position): where the actor is located in 2D space
        has_box (bool): if the actor is currently holding a box
    
    """
    def __init__(self, type: ActorType, position: Position = Position(0, 0)) -> None:
        """Initializes actor"""
        self.type = type
        self.position = position
        self.has_box = False
        self.md_sum = {}
        self.num_updates = {}
    

class State:
    """Class that represents a state in time of an Environment and Actors on it

    Attributes:
        dropoff_pos (list[Position]): the Positions where dropoff locations are, same index as dropoff_data
        dropoff_data (list[Space]): the Space data of each dropoff location, same index as dropoff_pos
        pickup_pos (list[Position]): the Positions where pickup locations are, same index as pickup_data
        pickup_data (list[Space]): the Space data of each pickup location, same index as pickup_pos
        actor_pos (list[Position]): the Positions of where each Actor is
        actor_dist (dict[Actor, dict[Actor, int]]): the distances each Actor is from one another. 
            Accessed by actor_dist[ActorType.RED][ActorType.BLUE], or actor_dist[ActorType.BLUE][ActorType.RED]
        actors (list[Actor]): the List of Actors in the State
    """

    def __init__(self, env: Environment, actors: list[Actor]) -> None:
        """Initializes the State"""
        self.dropoff_pos = []
        self.dropoff_data = []
        self.pickup_pos = []
        self.pickup_data = []
        self.actor_pos = {}
        self.actors = []
        self._init_env_locations(env)
        self._init_actors(actors)
        self.env = env
    
    def is_terminal(self) -> bool:
        """Checks if the State is considered terminal.
        
        A state is considered terminal if no actors are holding boxes,
            there are no more boxes left in the pick up locations,
            and all dropoff locations are filled to the maximum
            number of boxes.

        """
        # if an actor is holding a box
        for a in self.actors:
            if a.has_box:
                return False

        # if there are boxes still left to pickup
        for p in self.pickup_data:
            if p.num_blocks > 0:
                return False
        
        # if the dropoff location hasnt been filled with boxes
        for d in self.dropoff_data:
            if d.type != SpaceType.EMPTY:
                return False
        
        return True

    def _init_env_locations(self, env: Environment) -> None:
        """Initializes the dropoff and pickup positions/data"""
        for i in range(env.n):
            for j in range(env.m):
                space = env.at(Position(i, j))
                if space.type == SpaceType.DROP_OFF:
                    self.dropoff_pos.append(Position(i, j))
                    self.dropoff_data.append(space)
                elif space.type == SpaceType.PICK_UP:
                    self.pickup_pos.append(Position(i, j))
                    self.pickup_data.append(space)

    def _init_actors(self, actors: list[Actor]) -> None:
        """Initializes the actors and their data"""
        for actor in actors:
            self.actors.append(actor)
            self.actor_pos[actor] = (actor.position)

        for from_type in self.actors:
            for to_type in self.actors:
                if from_type == to_type: 
                    continue

                from_pos = self.actor_pos[from_type]
                to_pos = self.actor_pos[to_type]
                manhattan_dist = abs(to_pos.x - from_pos.x) + abs(to_pos.y - from_pos.y)

                from_type.md_sum[to_type] = manhattan_dist
                from_type.num_updates[to_type] = 0

    def update(self, actor, pos: Position, action):
        self.actor_pos[actor] = pos

        for other in self.actor_pos:
            if other == actor:
                continue

            actor.md_sum[other] += abs(pos.x - self.actor_pos[other].x) + abs(pos.y - self.actor_pos[other].y)
            actor.num_updates[other] += 1
                

        space = self.env.at(pos)
        
        if space.type == SpaceType.PICK_UP and action == Direction.PICKUP:
            actor.has_box = True
            blocksLeft = space.num_blocks - 1

            i = self.pickup_data.index(space)
            if blocksLeft <= 0:
                self.env.set(pos, Space(SpaceType.EMPTY))
            else:  
                self.env.set(pos, Space(SpaceType.PICK_UP, num_blocks= blocksLeft))

            self.pickup_data[i] = self.env.at(pos)
            space = self.env.at(pos)


        elif space.type == SpaceType.DROP_OFF and action == Direction.DROPOFF:
            actor.has_box = False
            blocksIn = space.num_blocks + 1

            i = self.dropoff_data.index(space)
            if blocksIn >= 5:
                self.env.set(pos, Space(SpaceType.EMPTY))
            else:  
                self.env.set(pos, Space(SpaceType.DROP_OFF, num_blocks= blocksIn))

            self.dropoff_data[i] = self.env.at(pos)
            space = self.env.at(pos)
    
    def __str__(self) -> str:
        """Convert to a string representing the dropoff locations, pickup locations,
            then actor data."""
        res = "["
        for i in range(len(self.dropoff_pos)):
            if i != 0:
                res += ","
            res += "(" + str(self.dropoff_pos[i]) + "," + str(self.dropoff_data[i]) + ")"

        res += "],\n["
        for i in range(len(self.pickup_pos)):
            if i != 0:
                res += ","
            res += "(" + str(self.pickup_pos[i]) + "," + str(self.pickup_data[i]) + ")"

        res += "],\n["
        add_comma = False
        for k, v in self.actor_pos.items():
            if add_comma:
                res += ","
            res += "(" + str(k) + "," + str(v) + "),"
            add_comma = True

        res += "]"
        return res

class QTable:
    """Class for QTable representation"""

    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(float))

    def get_q(self, state, action):
        state_index = (state.x, state.y)
        if state_index in self.table and action in self.table[state]:
            return self.table[state_index][action]
        else:
            return 0.0

    def set_q(self, state, action, value):
        state_index = (state.x, state.y)
        self.table[state_index][action] = value

    def get_max_q(self, state):
        state_index = (state.x, state.y)
        actions = self.table.get(state_index, {})
        if actions:
            return max(actions.values())
        else:
            return 0.0        

    def get_max_action(self, state):
        state_index = (state.x, state.y)
        actions = []
        max_q = float('-inf')

        for action in self.table[state_index]:
            if self.table[state_index][action] == max_q:
                max_q = self.table[state_index][action]
                actions.append(action)
            elif self.table[state_index][action] > max_q:
                actions.clear()
                max_q = self.table[state_index][action]
                actions.append(action)
        return actions

    def print_table(self, filename="q-table.csv"):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State', 'Action', 'Q-Value'])
            for state, actions in self.table.items():
                for action, q_value in actions.items():
                    writer.writerow([state, action, q_value])


def transition(current_pos: Position, action: Direction) -> Position:
    """Generates position based on action and current position"""
    # Assuming a grid world with coordinates (x, y)
    x = current_pos.x
    y = current_pos.y

    # Update the state based on the action
    if action == Direction.NORTH:  # Move up
        next_pos = Position(x + 1, y)
    elif action == Direction.SOUTH:  # Move down
        next_pos = Position(x - 1, y)
    elif action == Direction.WEST:  # Move left
        next_pos = Position(x, y - 1)
    elif action == Direction.EAST:  # Move right
        next_pos = Position(x, y + 1)
    elif action == Direction.PICKUP or action == Direction.DROPOFF or action == Direction.IDLE: #Pick up
        next_pos = Position(x,y)
    else:
        raise ValueError("Invalid action")
    
    # Check if the next state is valid (within the grid boundaries)
    # You might want to handle edge cases differently (e.g., if the agent reaches a wall)
    # Here, let's assume an infinite grid where all states are valid
    return next_pos

def model(current_state: State, actor, next_pos, action, policy, q_table):
    """Generates the next action to take depending on state and policy"""
    actors_list = []
    for actors in current_state.actors:
        actor_copy = deepcopy(actors)
        pos_copy = deepcopy(current_state.actor_pos[actors])
        actors_list.append(Actor(actor_copy.type, pos_copy))

    env_copy = deepcopy(current_state.env)
    next_state = State(env_copy, actors_list)
    
    for actors in next_state.actors:
        if actors.type == actor.type:
            next_actor = actors
            if actor.has_box == True:
                next_actor.has_box = True

    next_state.update(next_actor, next_pos, action)
    next_action, _ = policy(next_state, next_state.env, next_actor, q_table)
    return next_action


def Q_learning(action: Direction, actions: list[Direction], Q_table, actor, policy, gamma, alpha, current_state: State):
    """Executes Q learning reinforcement training"""
    current_pos = current_state.actor_pos[actor]
    current_q = Q_table.get_q(current_pos, action)

    next_pos = transition(current_pos, action)
    space = current_state.env.at(Position(next_pos.x, next_pos.y))

    reward = space.reward

    max_q_next = 0
    for a in actions:
        a_q = Q_table.get_q(next_pos, a)
        if a_q > max_q_next:
            max_q_next = a_q # Q(s+1, a)

    new_q_value = (1 - alpha) * current_q + alpha * (reward + gamma * max_q_next)
    Q_table.set_q(current_pos, action, new_q_value)

    current_state.update(actor, next_pos, action)
    

def SARSA(action: Direction, Q_table, actor: Actor, policy, gamma, alpha, current_state: State):
    """Executes SARSA reinforcement training"""
    current_pos = current_state.actor_pos[actor]
    current_q = Q_table.get_q(current_pos, action) # Q(S, A)
    
    next_pos = transition(current_pos, action) # S'
    space = current_state.env.at(Position(next_pos.x, next_pos.y))

    reward = space.reward

    next_action = model(current_state, actor, next_pos, action, policy, Q_table) # Choose A' from S'
    next_q = Q_table.get_q(next_pos, next_action) # Q(S', A')
    
    set_q = current_q + alpha * (reward + gamma * next_q - current_q)
    Q_table.set_q(current_pos, action, set_q) 

    current_state.update(actor, next_pos, action)
    return next_action

    
class Policy:
    """Class for generating policies that agents will folow"""

    def PRANDOM(self, current_state: State, env: Environment, actor, qTable) -> Direction:
        """Generates the RANDOM policy"""
        operator = []

        #stores actor pos as a space
        checkX = current_state.actor_pos[actor].x
        checkY = current_state.actor_pos[actor].y

        checkSpace = env.at(Position(checkX, checkY))
        #checks if actor can pick up, if so actor picks up box and environment is updated
        if self.checkPickUp(current_state, checkSpace, actor) == True:
            action = Direction.PICKUP
            operator.append(Direction.PICKUP)
        #checks if actor can drop off, if so actor drops off box and environment is updated
        elif self.checkDropOff(current_state, checkSpace, actor) == True:
            action = Direction.DROPOFF
            operator.append(Direction.DROPOFF)
        else:
            operator = self.valid_actions(current_state, current_state.actor_pos[actor], actor)

            if len(operator) == 0:
                action = Direction.IDLE
                operator.append(action)
            else:
                action = random.choice(operator)
                operator.append(action)

        return action, operator

    def PGREEDY(self, current_state: State, env: Environment, actor: Actor, qTable: QTable) -> Direction:
        """Generates the GREEDY policy"""
        operator = []

        checkX = current_state.actor_pos[actor].x
        checkY = current_state.actor_pos[actor].y
        checkSpace = env.at(Position(checkX, checkY))
        
        #checks if actor can pick up, if so actor picks up box and environment is updated
        if self.checkPickUp(current_state, checkSpace, actor) == True:
            action = Direction.PICKUP
            operator.append(Direction.PICKUP)
        #checks if actor can drop off, if so actor drops off box and environment is updated
        elif self.checkDropOff(current_state, checkSpace, actor) == True:
            action = Direction.DROPOFF
            operator.append(Direction.DROPOFF)
        else:
            #choose random max q value as move
            operator = self.valid_actions(current_state, current_state.actor_pos[actor], actor)
            best_actions = qTable.get_max_action(current_state.actor_pos[actor])

            if len(operator) != 0 and len(best_actions) != 0:
                for actions in best_actions:
                    if actions not in operator:
                        best_actions.remove(actions)

                if len(best_actions) != 0:
                    action = random.choice(best_actions)
                else:
                    action = random.choice(operator)
            else:
                action = Direction.IDLE
                operator.append(action)
            
        return action, operator
    
    def PEXPLOIT(self, current_state: State, env: Environmnt, actor, qTable: QTable) -> list[Direction]:
        """Generates the EXPLOIT policy"""
        operator = []

        checkX = current_state.actor_pos[actor].x
        checkY = current_state.actor_pos[actor].y

        checkSpace = env.at(Position(checkX, checkY))
        #checks if actor can pick up, if so actor picks up box and environment is updated
        if self.checkPickUp(current_state, checkSpace, actor) == True:
            action = Direction.PICKUP
            operator.append(Direction.PICKUP)
        #checks if actor can drop off, if so actor drops off box and environment is updated
        elif self.checkDropOff(current_state, checkSpace, actor) == True:
            action = Direction.DROPOFF
            operator.append(Direction.DROPOFF)
        else:
            operator = self.valid_actions(current_state, current_state.actor_pos[actor], actor)
            best_actions = qTable.get_max_action(current_state.actor_pos[actor])

            for actions in best_actions[:]:
                if actions not in operator:
                    best_actions.remove(actions)

            # Randomly chooses between max q value action(.8) or valid random action(.2)
            if(random.random() > 0.2 and len(best_actions) != 0):
                action = random.choice(best_actions)
            else:
                if len(operator) != 0:
                    action = random.choice(operator)
                else:
                    action = Direction.IDLE


        return action, operator
    
    def valid_actions(self, current_state: State, position, actor: Actor) -> list[Direction]:
        """Generates valid actions based on state and actor"""
        actions = []

        if self.check_position(current_state, (position.x+1, position.y), actor) == True: # NORTH
            actions.append(Direction.NORTH)
        
        if self.check_position(current_state, (position.x-1, position.y), actor) == True: # SOUTH
            actions.append(Direction.SOUTH)
        
        if self.check_position(current_state, (position.x, position.y+1), actor) == True: # EAST
            actions.append(Direction.EAST)

        if self.check_position(current_state, (position.x, position.y-1), actor) == True: # WEST
            actions.append(Direction.WEST)

        return actions
        
    def check_position(self, current_state: State, pos: Position, actor: Actor) -> bool:
        """If position is valid and unoccupied, return true"""

        #Checks if valid position, if invalid return false. 
        if pos[0] < 0 or pos[1] < 0:
            return False
        
        if pos[0] >= 5 or pos[1] >= 5:
            return False
        
        #Checks if position is occupied, if so returns false.
        for others in current_state.actors:
            if others == actor:
                continue
            #checks if x and y values matches other actors, if so returns false
            if current_state.actor_pos[others].x == pos[0] and current_state.actor_pos[others].y == pos[1]:
                return False
        
        return True
    
    
    def checkPickUp(self,current_state: State, checkSpace: Space, actor: Actor) -> bool:
        """Checks if actor can pick up box"""  
        if checkSpace.type == SpaceType.PICK_UP and actor.has_box == False and checkSpace.num_blocks > 0:
            return True
        return False
    
    def checkDropOff(self,current_state: State, checkSpace: Space, actor: Actor) -> bool: 
        """Checks if actor can drop off box"""   
        if checkSpace.type == SpaceType.DROP_OFF and actor.has_box == True and checkSpace.num_blocks < 5:
            return True
        return False
    
class Run:
    """Class to run the reinforcement training"""
    def __init__(self, init_state: State, table: QTable) -> None:
        self.init_state = init_state
        self.table = table

    def explore_q(self, policy, gamma, alpha) -> QTable:
        print("\nStarting Q-Learning explore...")
        num_terms = 0
        current_state = deepcopy(self.init_state)
        for i in range(500):
            # If state is terminal, reset PD-world but not Q-Table
            if current_state.is_terminal() == True:
                print("Terminal state reached in explore_q.")
                num_terms += 1
                current_state = deepcopy(self.init_state)
        
            for actor in current_state.actors:
                action, operators = policy(current_state, current_state.env, actor, self.table)
                Q_learning(action, operators, self.table, actor, policy, gamma, alpha, current_state)

        print("Number of times terminal state was reached: ", num_terms)

        print(current_state.actors[0].has_box, current_state.actors[1].has_box, current_state.actors[2].has_box)
        print(current_state.actor_pos[current_state.actors[0]], current_state.actor_pos[current_state.actors[1]], current_state.actor_pos[current_state.actors[2]])
        print(current_state.dropoff_data[0].num_blocks, current_state.dropoff_data[1].num_blocks, current_state.dropoff_data[2].num_blocks)
        print(current_state.pickup_data[0].num_blocks, current_state.pickup_data[1].num_blocks, current_state.pickup_data[2].num_blocks)
        self.print_avg_md(current_state)

        return self.table, current_state
    
    def explore_sarsa(self, policy, gamma, alpha) -> QTable:
        print("\nStarting SARSA explore...")
        num_terms = 0
        current_state = deepcopy(self.init_state)
        next_action = {}

        for actor in current_state.actors:
            next_action[actor], operators = policy(current_state, current_state.env, actor, self.table)

        for i in range(500):
            if current_state.is_terminal() == True:
                print("Terminal state reached in explore_sarsa.")
                num_terms += 1
                current_state = deepcopy(self.init_state)
                next_action.clear()
                for actor in current_state.actors:
                    next_action[actor], operators = policy(current_state, current_state.env, actor, self.table)

            for actor in current_state.actors:
                action = SARSA(next_action[actor], self.table, actor, policy, gamma, alpha, current_state)
                next_action[actor] = action

        print("Number of times terminal state was reached: ", num_terms)

        print(current_state.actors[0].has_box, current_state.actors[1].has_box, current_state.actors[2].has_box)
        print(current_state.actor_pos[current_state.actors[0]], current_state.actor_pos[current_state.actors[1]], current_state.actor_pos[current_state.actors[2]])
        print(current_state.dropoff_data[0].num_blocks, current_state.dropoff_data[1].num_blocks, current_state.dropoff_data[2].num_blocks)
        print(current_state.pickup_data[0].num_blocks, current_state.pickup_data[1].num_blocks, current_state.pickup_data[2].num_blocks)
        self.print_avg_md(current_state)

        return self.table, current_state

    def train_q(self, state, policy, gamma, alpha, exper) -> QTable:
        print("\nStarting Q-Learning train...")
        num_terms = 0
        current_state = deepcopy(state)
        for i in range(8500):
            if current_state.is_terminal() == True:
                print("Terminal state reached in", exper)
                num_terms += 1
                current_state = deepcopy(self.init_state)

            for actor in current_state.actors:
                action, operators = policy(current_state, current_state.env, actor, self.table)
                Q_learning(action, operators, self.table, actor, policy, gamma, alpha, current_state)    
        
        print("Number of times terminal state was reached: ", num_terms)

        self.print_avg_md(current_state)
                
        return self.table
    
    def train_sarsa(self, state, policy, gamma, alpha, exper) -> QTable:
        print("\nStarting SARSA train...")
        num_terms = 0
        current_state = deepcopy(state)
        next_action = {}

        for actor in current_state.actors:
            next_action[actor], _ = policy(current_state, current_state.env, actor, self.table)

        for i in range(8500):
            if current_state.is_terminal() == True:
                print("Terminal state reached in", exper)
                num_terms += 1
                current_state = deepcopy(self.init_state)
                next_action.clear()
                for actor in current_state.actors:
                    next_action[actor], _ = policy(current_state, current_state.env, actor, self.table)

            for actor in current_state.actors:
                action = SARSA(next_action[actor], self.table, actor, policy, gamma, alpha, current_state)
                next_action[actor] = action
        
        print("Number of times terminal state was reached: ", num_terms)

        self.print_avg_md(current_state)
         
        return self.table   
    
    def train_sarsa_change(self, state, policy, gamma, alpha, exper) -> QTable:
        print("\nStarting SARSA train...")
        num_terms = 0
        current_state = deepcopy(state)
        next_action = {}

        for actor in current_state.actors:
            next_action[actor], _ = policy(current_state, current_state.env, actor, self.table)

        for i in range(8500):
            if current_state.is_terminal() == True:
                print("Terminal state reached in", exper)
                num_terms += 1
                current_state = self.change_env(current_state)
                next_action.clear()
                for actor in current_state.actors:
                    next_action[actor], _ = policy(current_state, current_state.env, actor, self.table)

            for actor in current_state.actors:
                action = SARSA(next_action[actor], self.table, actor, policy, gamma, alpha, current_state)
                next_action[actor] = action

        print("Number of times terminal state was reached: ", num_terms)

        self.print_avg_md(current_state)
         
        return self.table          

    def change_env(self, current_state: State) -> State:
        actors_list = []
        for actors in current_state.actors:
            actor_copy = deepcopy(actors)
            pos_copy = deepcopy(current_state.actor_pos[actors])
            actors_list.append(Actor(actor_copy.type, pos_copy))

        env = Environment(n=5, m=5)
        env.set(Position(0, 0), Space(SpaceType.DROP_OFF))
        env.set(Position(0, 2), Space(SpaceType.DROP_OFF))
        env.set(Position(4, 3), Space(SpaceType.DROP_OFF))
        env.set(Position(4, 2), Space(SpaceType.PICK_UP, num_blocks=5))
        env.set(Position(3, 3), Space(SpaceType.PICK_UP, num_blocks=5))
        env.set(Position(2, 4), Space(SpaceType.PICK_UP, num_blocks=5))                

        new_state = State(env, actors_list)

        for actor in current_state.actors:
            for actor_copies in new_state.actors:
                if actor.type == actor_copies.type:
                    actor_copies.has_box = actor.has_box
        
        return new_state
    
    def print_avg_md(self, current_state):
        for actor in current_state.actors:
            print("\nActor: ", actor.type)
            for other in current_state.actors:
                if other == actor:
                    continue

                print("Average manhattan distance from ", other.type, ": ", actor.md_sum[other]/actor.num_updates[other])
    
def main():

    # Initialize environment
    env = Environment(n=5, m=5)
    env.set(Position(0, 0), Space(SpaceType.DROP_OFF))
    env.set(Position(2, 0), Space(SpaceType.DROP_OFF))
    env.set(Position(3, 4), Space(SpaceType.DROP_OFF))
    env.set(Position(0, 4), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(1, 3), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(4, 1), Space(SpaceType.PICK_UP, num_blocks=5))
    print(env)
    
    Red = Actor(ActorType.RED, Position(2, 2))
    Blue =  Actor(ActorType.BLUE, Position(2, 4))
    Black = Actor(ActorType.BLACK, Position(2, 0))
    actors = [Red, Blue, Black]
    state = State(env, actors)
    print(state)

    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 1.1                                #
    #                                                                              #
    ################################################################################

    print("\n---------Experiment 1.1---------")

    seed = 11
    np.random.seed(seed)
    random.seed(seed)

    table_q_1_1 = QTable()
    policy_q_1_1 = Policy()

    print("\nRun Q-Learning PRANDOM for 500 steps")

    # Run QLearning PRANDOM for 500 steps
    exper_1 = Run(state, table_q_1_1)
    explore_q_1, current_q_1 = exper_1.explore_q(policy_q_1_1.PRANDOM, gamma = 0.5, alpha = 0.3)
    explore_q_1.print_table("1_1_explore.csv")

    print("\nRun Q-Learning PRANDOM for remaining 8500 steps")

    # 1a. Run QLearning PRANDOM for remaining 8500 steps
    table_1a_1 = deepcopy(explore_q_1)
    train_1a_1 = Run(state, table_1a_1)
    result_1a_1 = train_1a_1.train_q(current_q_1, policy_q_1_1.PRANDOM, gamma = 0.5, alpha = 0.3, exper="1a")
    result_1a_1.print_table("1a_1.csv")

    print("\nRun Q-Learning PGREEDY for remaining 8500 steps")

    # 1b. Run QLearning PGREEDY for remaining 8500 steps
    table_1b_1 = deepcopy(explore_q_1)
    train_1b_1 = Run(state, table_1b_1)
    result_1b_1 = train_1b_1.train_q(current_q_1, policy_q_1_1.PGREEDY, gamma = 0.5, alpha = 0.3, exper="1b")
    result_1b_1.print_table("1b_1.csv")

    print("\nRun Q-Learning PEXPLOIT for remaining 8500 steps")

    # 1c. Run QLearning PEXPLOIT for remaining 8500 steps
    table_1c_1 = deepcopy(explore_q_1)
    train_1c_1 = Run(state, table_1c_1)
    result_1c_1 = train_1c_1.train_q(current_q_1, policy_q_1_1.PEXPLOIT, gamma = 0.5, alpha = 0.3, exper="1c")
    result_1c_1.print_table("1c_1.csv")

    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 1.2                                #
    #                                                                              #
    ################################################################################
    
    print("\n---------Experiment 1.2---------")

    seed = 12
    np.random.seed(seed)
    random.seed(seed)

    table_q_1_2 = QTable()
    policy_q_1_2 = Policy()

    print("\nRun Q-Learning PRANDOM for 500 steps")

    # Run QLearning PRANDOM for 500 steps
    exper_2 = Run(state, table_q_1_2)
    explore_q_2, current_q_2 = exper_2.explore_q(policy_q_1_2.PRANDOM, gamma = 0.5, alpha = 0.3)
    explore_q_2.print_table("1_2_explore.csv")

    print("\nRun Q-Learning PRANDOM for remaining 8500 steps")

    # 1a. Run QLearning PRANDOM for remaining 8500 steps
    table_1a_2 = deepcopy(explore_q_2)
    train_1a_2 = Run(state, table_1a_2)
    result_1a_2 = train_1a_2.train_q(current_q_2, policy_q_1_2.PRANDOM, gamma = 0.5, alpha = 0.3, exper="1a")
    result_1a_2.print_table("1a_2.csv")

    print("\nRun Q-Learning PGREEDY for remaining 8500 steps")

    # 1b. Run QLearning PGREEDY for remaining 8500 steps
    table_1b_2 = deepcopy(explore_q_2)
    train_1b_2 = Run(state, table_1b_2)
    result_1b_2 = train_1b_2.train_q(current_q_2, policy_q_1_2.PGREEDY, gamma = 0.5, alpha = 0.3, exper="1b")
    result_1b_2.print_table("1b_2.csv")

    print("\nRun Q-Learning PEXPLOIT for remaining 8500 steps")

    # 1c. Run QLearning PEXPLOIT for remaining 8500 steps
    table_1c_2 = deepcopy(explore_q_2)
    train_1c_2 = Run(state, table_1c_2)
    result_1c_2 = train_1c_2.train_q(current_q_2, policy_q_1_2.PEXPLOIT, gamma = 0.5, alpha = 0.3, exper="1c")
    result_1c_2.print_table("1c_1.csv")


    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 2.1                                #
    #                                                                              #
    ################################################################################
    
    print("\n---------Experiment 2.1---------")

    seed = 21
    np.random.seed(seed)
    random.seed(seed)

    print("\nRun Q-Learning PRANDOM for 500 steps")

    # Run SARSA PRANDOM for 500 steps
    policy_s_1 = Policy()
    table_sarsa_1 = QTable()

    exper_sarsa_1 = Run(state, table_sarsa_1)
    explore_sarsa_1, current_sarsa_1 = exper_sarsa_1.explore_sarsa(policy_s_1.PRANDOM, gamma = 0.5, alpha = 0.3)
    explore_sarsa_1.print_table("2_1_explore.csv")

    print("\nRun SARSA PEXPLOIT for remaining 8500 steps")

    # Run SARSA PEXPLOIT for remaining 8500 steps
    table_2_1 = deepcopy(explore_sarsa_1)
    train_2_1 = Run(state, table_2_1)
    result_2_1 = train_2_1.train_sarsa(current_sarsa_1, policy_s_1.PEXPLOIT, gamma = 0.5, alpha = 0.3, exper="2")
    result_2_1.print_table("2_1_result.csv")


    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 2.2                                #
    #                                                                              #
    ################################################################################
    
    print("\n---------Experiment 2.2---------")

    seed = 22
    np.random.seed(seed)
    random.seed(seed)

    print("\nRun SARSA PRANDOM for 500 steps")

    # Run SARSA PRANDOM for 500 steps
    policy_s_2 = Policy()
    table_sarsa_2 = QTable()

    exper_sarsa_2 = Run(state, table_sarsa_2)
    explore_sarsa_2, current_sarsa_2 = exper_sarsa_2.explore_sarsa(policy_s_2.PRANDOM, gamma = 0.5, alpha = 0.3)
    explore_sarsa_2.print_table("2_2_explore.csv")

    print("\nRun SARSA PEXPLOIT for remaining 8500 steps")

    # Run SARSA PEXPLOIT for remaining 8500 steps
    table_2_2 = deepcopy(explore_sarsa_2)
    train_2_2 = Run(state, table_2_2)
    result_2_2 = train_2_2.train_sarsa(current_sarsa_2, policy_s_2.PEXPLOIT, gamma = 0.5, alpha = 0.3, exper="2")
    result_2_2.print_table("2_2_result.csv")    

    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 3.1                                #
    #                                                                              #
    ################################################################################
    
    print("\n---------Experiment 3.1---------")

    seed = 31
    np.random.seed(seed)
    random.seed(seed)

    print("\nRun SARSA PRANDOM for 500 steps, alpha = 0.15")

    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.15
    policy_s3_115 = Policy()
    table_sarsa3_115 = QTable()

    exper_sarsa3_115 = Run(state, table_sarsa3_115)
    explore_sarsa3_115, current_sarsa3_115 = exper_sarsa3_115.explore_sarsa(policy_s3_115.PRANDOM, gamma = 0.5, alpha = 0.15)
    explore_sarsa3_115.print_table("3_1_a15_explore.csv")    

    print("\nRun SARSA PEXPLOIT for remaining 8500 steps, alpha = 0.15")

    table_3_1_15 = deepcopy(explore_sarsa3_115)
    train_3_1_15 = Run(state, table_3_1_15)
    result_3_1_15 = train_3_1_15.train_sarsa(current_sarsa3_115, policy_s3_115.PEXPLOIT, gamma = 0.5, alpha = 0.15, exper="3")
    result_3_1_15.print_table("3_1_result_a15.csv")

    print("\nRun SARSA PRANDOM for 500 steps, alpha = 0.45")

    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.45
    policy_s3_145 = Policy()
    table_sarsa3_145 = QTable()

    exper_sarsa3_145 = Run(state, table_sarsa3_145)
    explore_sarsa3_145, current_sarsa3_145 = exper_sarsa3_145.explore_sarsa(policy_s3_145.PRANDOM, gamma = 0.5, alpha = 0.15)
    explore_sarsa3_145.print_table("3_1_a45_explore.csv")  
    

    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.45

    print("\nRun SARSA PEXPLOIT for remaining 8500 steps, alpha = 0.45")

    table_3_1_45 = deepcopy(explore_sarsa3_145)
    train_3_1_45 = Run(state, table_3_1_45)
    result_3_1_45 = train_3_1_45.train_sarsa(current_sarsa3_145, policy_s3_145.PEXPLOIT, gamma = 0.5, alpha = 0.45, exper="3")
    result_3_1_45.print_table("3_1_result_a45.csv")

    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 3.2                                #
    #                                                                              #
    ################################################################################
    
    print("\n---------Experiment 3.2---------")

    seed = 3215
    np.random.seed(seed)
    random.seed(seed)

    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.15

    print("\nRun SARSA PRANDOM for 500 steps, alpha = 0.15")

    policy_s3_215 = Policy()
    table_sarsa3_215 = QTable()

    exper_sarsa3_215 = Run(state, table_sarsa3_215)
    explore_sarsa3_215, current_sarsa3_215 = exper_sarsa3_215.explore_sarsa(policy_s3_215.PRANDOM, gamma = 0.5, alpha = 0.15)
    explore_sarsa3_215.print_table("3_2_explore.csv")    

    print("\nRun SARSA PEXPLOIT for remaining 8500 steps, alpha = 0.15")

    table_3_2_15 = deepcopy(explore_sarsa_2)
    train_3_2_15 = Run(state, table_3_2_15)
    result_3_2_15 = train_3_2_15.train_sarsa(current_sarsa3_215, policy_s3_215.PEXPLOIT, gamma = 0.5, alpha = 0.15, exper="3")
    result_3_2_15.print_table("3_2_result_a15.csv")

    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.45

    print("\nRun SARSA PRANDOM for 500 steps, alpha = 0.45")

    policy_s3_245 = Policy()
    table_sarsa3_245 = QTable()

    exper_sarsa3_245 = Run(state, table_sarsa3_245)
    explore_sarsa3_245, current_sarsa3_245 = exper_sarsa3_245.explore_sarsa(policy_s3_245.PRANDOM, gamma = 0.5, alpha = 0.15)
    explore_sarsa3_245.print_table("3_2_a45_explore.csv")  

    
    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.45

    print("\nRun SARSA PEXPLOIT for remaining 8500 steps, alpha = 0.45")

    table_3_2_45 = deepcopy(explore_sarsa3_245)
    train_3_2_45 = Run(state, table_3_2_45)
    result_3_2_45 = train_3_2_45.train_sarsa(current_sarsa3_245, policy_s3_245.PEXPLOIT, gamma = 0.5, alpha = 0.45, exper="3")
    result_3_2_45.print_table("3_2_result_a45.csv")

    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 4.1                                #
    #                                                                              #
    ################################################################################
    
    print("\n---------Experiment 4.1---------")

    seed = 41
    np.random.seed(seed)
    random.seed(seed)

    print("Run SARSA PRANDOM for 500 steps")

    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.3 and gamma = 0.5
    policy_s4_1 = Policy()
    table_sarsa4_1 = QTable()

    exper_sarsa4_1 = Run(state, table_sarsa4_1)
    explore_sarsa4_1, current_sarsa4_1 = exper_sarsa4_1.explore_sarsa(policy_s4_1.PRANDOM, gamma = 0.5, alpha = 0.15)
    explore_sarsa4_1.print_table("4_1_explore.csv")    

    print("Run SARSA PEXPLOIT for remaining 8500 steps, changing pickup locations")

    # Change pickup locations after terminal state reached
    table_4_1 = deepcopy(explore_sarsa4_1)
    train_4_1 = Run(state, table_4_1)
    result_4_1 = train_4_1.train_sarsa_change(current_sarsa4_1, policy_s4_1.PEXPLOIT, gamma = 0.5, alpha = 0.15, exper="4")
    result_4_1.print_table("4_1_result.csv")

    ################################################################################
    #                                                                              #
    #                                EXPERIMENT 4.2                                #
    #                                                                              #
    ################################################################################
    
    print("\n---------Experiment 4.2---------")

    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    print("Run SARSA PRANDOM for 500 steps")

    # Rerun experiment 2 SARSA PEXPLOIT with alpha = 0.3 and gamma = 0.5
    policy_s4_2 = Policy()
    table_sarsa4_2 = QTable()

    exper_sarsa4_2 = Run(state, table_sarsa4_2)
    explore_sarsa4_2, current_sarsa4_2 = exper_sarsa4_2.explore_sarsa(policy_s4_2.PRANDOM, gamma = 0.5, alpha = 0.15)
    explore_sarsa4_2.print_table("4_2_explore.csv")    


    # Change pickup locations after terminal state reached

    print("Run SARSA PEXPLOIT for remaining 8500 steps, changing pickup locations")

    table_4_2 = deepcopy(explore_sarsa4_2)
    train_4_2 = Run(state, table_4_2)
    result_4_2 = train_4_2.train_sarsa_change(current_sarsa4_2, policy_s4_2.PEXPLOIT, gamma = 0.5, alpha = 0.15, exper="4")
    result_4_2.print_table("4_2_result.csv")


if __name__ == "__main__":
    main()

