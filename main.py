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
        for i in range(self.n):
            for j in range(self.m):
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
    

class State:
    """Class that represents a state in time of an Environment and Actors on it

    Attributes:
        dropoff_pos (list[Position]): the Positions where dropoff locations are, same index as dropoff_data
        dropoff_data (list[Space]): the Space data of each dropoff location, same index as dropoff_pos
        pickup_pos (list[Position]): the Positions where pickup locations are, same index as pickup_data
        pickup_data (list[Space]): the Space data of each pickup location, same index as pickup_pos
        actor_pos (list[Position]): the Positions of where each Actor is
        actor_dist (dict[ActorType, dict[ActorType, int]]): the distances each Actor is from one another. 
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
        self.actor_dist = {} # manhattan distance
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
            if d.num_blocks < d.max_blocks:
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

        for from_type, from_pos in self.actor_pos.items():
            for to_type, to_pos in self.actor_pos.items():
                if from_type == to_type: 
                    continue

                manhattan_dist = abs(to_pos.x - from_pos.x) + abs(to_pos.y - from_pos.y)
                if from_type not in self.actor_dist:
                    self.actor_dist[from_type] = {to_type: manhattan_dist}
                else:
                    self.actor_dist[from_type][to_type] = manhattan_dist

    def update(self, actor, pos: Position):
        self.actor_pos[actor] = pos

        for other in self.actor_pos:
            if other == actor:
                continue

            self.actor_dist[actor][other] = abs(pos.x - self.actor_pos[other].x) + abs(pos.y - self.actor_pos[other].y)

        space = self.env.at(pos)

        if space.type == SpaceType.PICK_UP:
            actor.has_box = True
            blocksLeft = space.num_blocks - 1
            if blocksLeft <= 0:
                self.env.set(pos, Space(SpaceType.EMPTY))
            else:  
                self.env.set(pos, Space(SpaceType.PICK_UP, num_blocks= blocksLeft))
        elif space.type == SpaceType.DROP_OFF:
            actor.has_box = False
            blocksIn = space.num_blocks + 1
            if blocksIn >= 5:
                self.env.set(pos, Space(SpaceType.EMPTY))
            else:  
                self.env.set(pos, Space(SpaceType.PICK_UP, num_blocks= blocksIn))
    
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
    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(float))

    def get_q(self, state, action):
        if state in self.table and action in self.table[state]:
            return self.table[state][action]
        else:
            return 0.0

    def set_q(self, state, action, value):
        self.table[state][action] = value

    def get_max_q(self, state):
        actions = self.table.get(state, {})
        if actions:
            return max(actions.values())
        else:
            return 0.0        

    def get_max_action(self, state):
        max_action = None
        max_q = float('-inf')
        for action, q_value in self.table[state].items():
            if q_value > max_q:
                max_q = q_value
                max_action = action
        return max_action

    def print_table(self, filename="q-table.csv"):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State', 'Action', 'Q-Value'])
            for state, actions in self.table.items():
                for action, q_value in actions.items():
                    writer.writerow([state, action, q_value])


def transition(current_pos: Position, action: Direction) -> Position:
    
    # Assuming a grid world with coordinates (x, y)
    x = current_pos.x
    y = current_pos.y

    # Update the state based on the action
    if action == Direction.NORTH:  # Move up
        next_pos = Position(x, y + 1)
    elif action == Direction.SOUTH:  # Move down
        next_pos = Position(x, y - 1)
    elif action == Direction.WEST:  # Move left
        next_pos = Position(x - 1, y)
    elif action == Direction.EAST:  # Move right
        next_pos = Position(x + 1, y)
    elif action == Direction.PICKUP or action == Direction.DROPOFF or action == Direction.IDLE: #Pick up
        next_pos = Position(x,y)
    else:
        raise ValueError("Invalid action")
    
    # Check if the next state is valid (within the grid boundaries)
    # You might want to handle edge cases differently (e.g., if the agent reaches a wall)
    # Here, let's assume an infinite grid where all states are valid
    return next_pos

def Q_learning(action: Direction, Q_table, actor, gamma, alpha, actions: list[Direction], current_state: State):

    current_pos = current_state.actor_pos[actor]
    space = current_state.env.at(Position(current_pos.x, current_pos.y))

    next_pos = transition(current_pos, action)
    
    if space.type == SpaceType.DROP_OFF and action == Direction.DROPOFF:
        reward = space.reward
        actor.has_box = False
    elif space.type == SpaceType.PICK_UP and action == Direction.PICKUP:
        reward = space.reward
        actor.has_box = True
    else:
        reward = space.reward

    prev_state = deepcopy(current_state)
    current_state.update(actor, next_pos) # New state

    current_q = Q_table.get_q(prev_state, action)
    max_q_next = Q_table.get_max_q(current_state)
    new_q_value = (1 - alpha) * current_q + alpha * (reward + gamma * max_q_next)
    Q_table.set_q(prev_state, action, new_q_value)

def SARSA(policy: list[Direction], transition: transition, Q_table, gamma, alpha, actions: list[Direction], initial_state: State):
    current_state = initial_state
    current_action = None
    
    for i, actor in current_state.actor_pos:
        current_pos = current_state.actor_pos[actor]
        space = current_state.env.at(Position(current_pos.x, current_pos.y))
        
        next_pos = transition(current_pos, policy[i])
        
        if space.type == SpaceType.DROP_OFF and policy[i] == Direction.DROPOFF:
            reward = space.reward
        elif space.type == SpaceType.PICK_UP  and policy[i] == Direction.PICKUP:
            reward = space.reward
        else:
            reward = space.reward
        if current_action is not None:
            next_action = current_action
            next_q = Q_table.get_q(next, next_action)
        else:
            next_action = None
            next_q = 0  # Terminal state

        current_q = Q_table.get_q(current_pos, policy[i])
        set_q = current_q + alpha * (reward + gamma * next_q - current_q)
        Q_table.set_q(current_pos, policy[i], set_q) 

        current_pos = next_pos
        current_action = next_action

    
class Policy:
    
    def PRANDOM(self, current_state: State, env: Environment, actor, qTable, gamma, alpha) -> Direction:
        operator = []

        #stores actor pos as a space
        checkX = current_state.actor_pos[actor].x
        checkY = current_state.actor_pos[actor].y

        checkSpace = env.at(Position(checkX, checkY))

        #checks if actor can pick up, if so actor picks up box and environment is updated
        if self.checkPickUp(current_state, checkSpace, actor) == True:
            actor.has_box = True
            action = Direction.PICKUP
        #checks if actor can drop off, if so actor drops off box and environment is updated
        elif self.checkDropOff(current_state, checkSpace, actor) == True:
            action = Direction.DROPOFF
        else:
            operator = self.valid_actions(current_state, current_state.actor_pos[actor], actor)

            if len(operator) == 0:
                action = Direction.IDLE
            else:
                action = random.choice(operator)

        return action, operator
    
    def PGREEDY(self, current_state: State, env: Environment, qTable, gamma, alpha) -> list[Direction]:
        operator = []
        i = 0
        for actor in current_state.actors:
            checkX = current_state.actor_pos[actor].x
            checkY = current_state.actor_pos[actor].y
            checkSpace = env.at(Position(checkX, checkY))
            #checks if actor can pick up, if so actor picks up box and environment is updated
            if self.checkPickUp(current_state, checkSpace, actor) == True:
                operator.append(Direction.PICKUP)
            #checks if actor can drop off, if so actor drops off box and environment is updated
            elif self.checkDropOff(current_state, checkSpace, actor) == True:
                operator.append(Direction.DROPOFF)
            else:
                #choose random max q value as move
                actions = self.valid_actions(current_state, current_state.actor_pos[actor], actor)
                max_q, best_action = qTable.get_max_action(Position(checkX, checkY))
                
                if best_action in actions:
                    action = best_action
                else:
                    action = random.choice(actions)
                
                operator.append(action)
            i += 1
        #Updates Q table using function, switch to sarsa function if running sarsa
        Q_learning(operator, transition, qTable, gamma, alpha, actions, current_state)
        return operator
    
    def PEXPLOIT(self, current_state: State, env: Environment, qTable, gamma, alpha) -> list[Direction]:
        operator = []
        i = 0
        for actor in current_state.actors:
            checkX = current_state.actor_pos[actor].x
            checkY = current_state.actor_pos[actor].y
            checkSpace = env.at(Position(checkX, checkY))
            #checks if actor can pick up, if so actor picks up box and environment is updated
            if self.checkPickUp(current_state, checkSpace, actor) == True:
                operator.append(Direction.PICKUP)
            #checks if actor can drop off, if so actor drops off box and environment is updated
            elif self.checkDropOff(current_state, checkSpace, actor) == True:
                operator.append(Direction.DROPOFF)
            else:
                #creates 2 lists, 1 for max q values, 1 for valid random actions   
                qActions = np.argmax(qTable[current_state.actor_pos[actor]])
                qAction = random.choice(qActions)
                actions = self.valid_actions(current_state, current_state.actor_pos[actor], actor)
                action = random.choice(actions)
                # Randomly chooses between max q value action(.8) or valid random action(.2)
                if(random.random() < 0.8):
                    operator.append(qAction)
                else:
                    operator.append(action)
            i += 1
        #Updates Q table using function, switch to sarsa function if running sarsa
        Q_learning(operator, transition, qTable, gamma, alpha, actions, current_state)
        return operator
    
    def valid_actions(self, current_state: State, position, actor: Actor) -> list[Direction]:
        actions = []

        if self.check_position(current_state, (position.x, position.y+1), actor) == True: # NORTH
            actions.append(Direction.NORTH)
        
        if self.check_position(current_state, (position.x, position.y-1), actor) == True: # SOUTH
            actions.append(Direction.SOUTH)
        
        if self.check_position(current_state, (position.x+1, position.y), actor) == True: # EAST
            actions.append(Direction.EAST)

        if self.check_position(current_state, (position.x-1, position.y), actor) == True: # WEST
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
    def __init__(self, init_state: State) -> None:
        self.init_state = init_state
        self.table = QTable()
        self.current_state = init_state

    def explore(self, gamma, alpha) -> QTable:
        policy = Policy()
        current_state = deepcopy(self.init_state)
        for i in range(500):
            # If state is terminal, reset PD-world but not Q-Table
            if current_state.is_terminal == True:
                print("Terminal state reached in explore.")
                current_state = deepcopy(self.init_state)

            for actor in current_state.actors:
                action, operators = policy.PRANDOM(current_state, current_state.env, actor, self.table, gamma, alpha)
                Q_learning(action, self.table, actor, gamma, alpha, operators, current_state)

        return self.table, current_state
    
    def greedy(self, table, current_state, gamma, alpha) -> QTable:
        policy = Policy()

        for i in range(8500):
            actions = []

            if current_state.is_terminal == True:
                print("Terminal state reached in greedy.")
                current_state = deepcopy(self.init_state)

            actions = policy.PGREEDY(current_state, current_state.env, table, gamma, alpha)
            current_state.update(actions)
        
        return table

    
def main():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    env = Environment(n=5, m=5)
    env.set(Position(0, 0), Space(SpaceType.DROP_OFF))
    env.set(Position(0, 2), Space(SpaceType.DROP_OFF))
    env.set(Position(4, 3), Space(SpaceType.DROP_OFF))
    env.set(Position(4, 0), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(3, 1), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(2, 4), Space(SpaceType.PICK_UP, num_blocks=5))
    print(env)
    
    Red = Actor(ActorType.RED, Position(2, 2))
    Blue =  Actor(ActorType.BLUE, Position(2, 4))
    Black = Actor(ActorType.BLACK, Position(2, 0))
    actors = [Red, Blue, Black]
    state = State(env, actors)
    print(state)

    exper = Run(state)
    explore, current_state = exper.explore(gamma = 0.5, alpha = 0.3)
    explore.print_table()

    #train_1a_state = deepcopy(current_state)
    #exper_1a = Run(state)
    #train_1a = exper_1a.greedy(explore, train_1a_state, gamma = 0.5, alpha = 0.3)
    #train_1a.print_q_table()

if __name__ == "__main__":
    main()

