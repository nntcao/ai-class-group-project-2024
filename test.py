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

    def test_dropoff_add(self):
        sp = Space(SpaceType.DROP_OFF)
        sp.add_block()
        self.assertEqual(sp.get_num_blocks(), 1)

    def test_pickup_remove(self):
        sp = Space(SpaceType.PICK_UP, 5, 5)
        sp.remove_block()
        self.assertEqual(sp.get_num_blocks(), 4)

    def test_pickup_invalid_remove(self):
        sp = Space(SpaceType.PICK_UP, 0, 5)
        self.assertRaises(ValueError, sp.remove_block)

    def test_dropoff_invalid_add(self):
        sp = Space(SpaceType.PICK_UP, 5, 5)
        self.assertRaises(ValueError, sp.add_block)


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


if __name__ == "__main__":
    unittest.main()
