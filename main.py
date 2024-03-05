from enum import Enum


class Position:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y


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

    def get_num_blocks(self):
        return self.num_blocks

    def remove_block(self) -> None:
        self.num_blocks -= 1
        if self.num_blocks < 0:
            raise ValueError("number of blocks cannot be less than 0 on space")

    def add_block(self) -> None:
        self.num_blocks += 1
        if self.num_blocks > self.max_blocks:
            raise ValueError("max blocks exceeded on space")


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
    def __init__(self, type: ActorType) -> None:
        self.type = type


class State:
    def __init__(self, env: Environment, actors: list[Actor]) -> None:
        pass


def main():
    env = Environment(n=5, m=5)
    env.set(Position(0, 0), Space(SpaceType.DROP_OFF))
    env.set(Position(2, 0), Space(SpaceType.DROP_OFF))
    env.set(Position(3, 4), Space(SpaceType.DROP_OFF))
    env.set(Position(0, 4), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(1, 3), Space(SpaceType.PICK_UP, num_blocks=5))
    env.set(Position(4, 1), Space(SpaceType.PICK_UP, num_blocks=5))
    print(env)


if __name__ == "__main__":
    main()
