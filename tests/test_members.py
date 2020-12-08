import unittest


class MyTestCase(unittest.TestCase):

    def test_members(self):
        with open('members.txt') as fh:
            lines = fh.readlines()

        self.assertTrue(lines[0].startswith('member 1: '))
        self.assertTrue(lines[1].startswith('member 2: '))
        self.assertTrue(lines[2].startswith('member 3: '))


if __name__ == '__main__':
    unittest.main()
