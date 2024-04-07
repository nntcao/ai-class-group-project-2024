import unittest
from main import *


class TestShape(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(Space(SpaceType.EMPTY).type, SpaceType.EMPTY)

    def test_dropoff(self):
        self.assertEqual(Space(SpaceType.DROP_OFF).type, SpaceType.DROP_OFF)

    def test_pickup(self):
        self.assertEqual(Space(SpaceType.PICK_UP).type, SpaceType.PICK_UP)

    def test_invalid_num_blocks(self):
        self.assertRaises(ValueError, Space, SpaceType.PICK_UP, -1, 5)

    def test_invalid_max_blocks(self):
        self.assertRaises(ValueError, Space, SpaceType.PICK_UP, 0, -1)

    def test_exceed_limit(self):
        self.assertRaises(ValueError, Space, SpaceType.PICK_UP, 16, 4)

    def test_empty_reward(self):
        self.assertEqual(Space(SpaceType.EMPTY).reward, -1)

    def test_dropoff_reward(self):
        self.assertEqual(Space(SpaceType.DROP_OFF).reward, 13)

    def test_pickup_reward(self):
        self.assertEqual(Space(SpaceType.PICK_UP).reward, 13)


class TestEnvironment(unittest.TestCase):
    def test_creation(self):
        env = Environment(n=1, m=1)
        self.assertEqual(env.at(Position(0, 0)).type, SpaceType.EMPTY)

    def test_creation_pickup(self):
        env = Environment(n=1, m=1)
        env.set(Position(0, 0), Space(SpaceType.PICK_UP))
        self.assertEqual(env.at(Position(0, 0)).type, SpaceType.PICK_UP)

    def test_creation_dropoff(self):
        env = Environment(n=1, m=1)
        env.set(Position(0, 0), Space(SpaceType.DROP_OFF))
        self.assertEqual(env.at(Position(0, 0)).type, SpaceType.DROP_OFF)


class TestState(unittest.TestCase):
    def test_is_not_terminal(self):
        env = Environment(n=5, m=5)
        env.set(Position(0, 0), Space(SpaceType.DROP_OFF))
        env.set(Position(2, 0), Space(SpaceType.DROP_OFF))
        env.set(Position(3, 4), Space(SpaceType.DROP_OFF))
        env.set(Position(0, 4), Space(SpaceType.PICK_UP, num_blocks=5))
        env.set(Position(1, 3), Space(SpaceType.PICK_UP, num_blocks=5))
        env.set(Position(4, 1), Space(SpaceType.PICK_UP, num_blocks=5))

        state = State(env, [Actor(ActorType.BLUE, Position(3, 2)), Actor(ActorType.RED, Position(0, 4))])
        self.assertEqual(state.is_terminal(), False)

    
    def test_is_terminal(self):
        env_terminal = Environment(n=1, m=2)
        env_terminal.set(Position(0, 0), Space(SpaceType.PICK_UP, 0, 5))
        env_terminal.set(Position(0, 1), Space(SpaceType.DROP_OFF, 5, 5))

        terminal_state = State(env_terminal, [])

        self.assertEqual(terminal_state.is_terminal(), True)


if __name__ == "__main__":
    unittest.main()
