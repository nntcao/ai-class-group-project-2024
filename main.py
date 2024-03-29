from enum import Enum

class Position:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return "(" + str(self.x) + ", " + str(self.y) + ")"


class SpaceType(Enum):
    EMPTY = "E"
    DROP_OFF = "D"
    PICK_UP = "P"


class Space:
    def __init__(self, type: SpaceType = SpaceType.EMPTY, num_blocks: int = 0, max_blocks: int = 5) -> None:
        if num_blocks > max_blocks:
            raise ValueError("max blocks exceeded on space")
        if num_blocks < 0:
            raise ValueError("num blocks cannot be less than 0 on space")
        if max_blocks < 0:
            raise ValueError("max blocks cannot be less than 0 on space")
        self.type = type
        self.num_blocks = num_blocks
        self.max_blocks = max_blocks
    
    def __str__(self) -> str:
        return "(" + str(self.type) + "," + str(self.num_blocks) + "," + str(self.max_blocks) + ")"

class Environment:
    def __init__(self, n: int = 5, m: int = 5) -> None:
        self.n = n
        self.m = m
        self.pd_world = [[Space(max_blocks=0)
                          for _ in range(m)] for _ in range(n)]

    def set(self, pos: Position, space: Space) -> None:
        if pos.x < 0 or self.n <= pos.x or pos.y < 0 or self.m <= pos.y:
            raise ValueError("cannot set space as position is out of bounds")
        self.pd_world[pos.x][pos.y] = space

    def at(self, pos: Position) -> Space:
        if pos.x < 0 or self.n <= pos.x or pos.y < 0 or self.m <= pos.y:
            raise ValueError("cannot find space as position is out of bounds")
        return self.pd_world[pos.x][pos.y]

    def within_bounds(self, pos: Position) -> None:
        return -1 < pos.x and pos.x < self.n and \
            -1 < pos.y and pos.y < self.m

    def __str__(self) -> str:
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


class ActorType(Enum):
    RED = "R"
    BLUE = "B"
    BLACK = "K"


class Actor:
    def __init__(self, type: ActorType, position: Position = Position(0, 0)) -> None:
        self.type = type
        self.position = position

class State:
    dropoff_pos: list[Position]
    dropoff_data: list[Space]
    pickup_pos: list[Position]
    pickup_data: list[Space]
    actor_pos: list[Position]
    actor_dist: dict[ActorType, dict[ActorType, int]]

    def __init__(self, env: Environment, actors: list[Actor]) -> None:
        self.dropoff_pos = []
        self.dropoff_data = []
        self.pickup_pos = []
        self.pickup_data = []
        self.actor_pos = {}
        self.actor_dist = {} # manhattan distance
        self._init_env_locations(env)
        self._init_actor_locations(actors)

    def _init_env_locations(self, env: Environment) -> None:
        for i in range(env.n):
            for j in range(env.m):
                space = env.at(Position(i, j))
                if space.type == SpaceType.DROP_OFF:
                    self.dropoff_pos.append(Position(i, j))
                    self.dropoff_data.append(space)
                elif space.type == SpaceType.PICK_UP:
                    self.pickup_pos.append(Position(i, j))
                    self.pickup_data.append(space)

    def _init_actor_locations(self, actors: list[Actor]) -> None:
        for actor in actors:
            self.actor_pos[actor.type] = (actor.position)

        for from_type, from_pos in self.actor_pos.items():
            for to_type, to_pos in self.actor_pos.items():
                if from_type == to_type: 
                    continue

                manhattan_dist = abs(to_pos.x - from_pos.x) + abs(to_pos.y - from_pos.y)
                if from_type not in self.actor_dist:
                    self.actor_dist[from_type] = {to_type: manhattan_dist}
                else:
                    self.actor_dist[from_type][to_type] = manhattan_dist
    
    def __str__(self) -> str:
        res = "["
        for i in range(len(self.dropoff_pos)):
            res += "(" + str(self.dropoff_pos[i]) + "," + str(self.dropoff_data[i]) + "),"

        res += "],\n["
        for i in range(len(self.pickup_pos)):
            res += "(" + str(self.pickup_pos[i]) + "," + str(self.pickup_data[i]) + "),"

        res += "],\n["
        for k, v in self.actor_pos.items():
            res += "(" + str(k) + "," + str(v) + "),"

        res += "]"
        return res

# class Policy:
#     def get_next_move(env: Environment, actors: list[Actor], q_table: QTable):
#         pass


def main():
    env = Environment(n=5, m=5)
    env.set(Position(0, 0), Space(SpaceType.DROP_OFF))
    env.set(Position(2, 0), Space(SpaceType.DROP_OFF))
    env.set(Position(3, 4), Space(SpaceType.DROP_OFF))
    env.set(Position(0, 4), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(1, 3), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(4, 1), Space(SpaceType.PICK_UP, num_blocks=5))
    print(env)

    state = State(env, [Actor(ActorType.BLUE, Position(3, 2)), Actor(ActorType.RED, Position(0, 4))])
    print(state)


if __name__ == "__main__":
    main()
