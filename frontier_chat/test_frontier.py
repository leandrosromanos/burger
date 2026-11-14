import math
import unittest
import numpy as np
from frontier_explorer import Grid, GoalHistory, dilate, flood, plan_frontiers
from prepare_nav2 import configure


def partial():
    a = np.full((70, 90), 100, dtype=np.int16)
    a[5:65, 5:45] = 0
    a[5:65, 45:85] = -1
    return a

class FrontierTests(unittest.TestCase):
    def test_frontier_safe_goals(self):
        grid = Grid(partial(), .05)
        goals, stats = plan_frontiers(grid, (1., 1.))
        self.assertTrue(goals)
        self.assertEqual(stats['reachable'], 1)
        for _, x, y, yaw in goals:
            row, col = grid.cell(x, y)
            self.assertEqual(grid.cells[row,col], 0)
            self.assertLess(x, 2.25-.35)
            self.assertTrue(abs(yaw) < .1)
    def test_closed_room_completion(self):
        a = np.full((70,70), -1, dtype=np.int16)
        a[5:65,5:65] = 100
        a[6:64,6:64] = 0
        goals, stats = plan_frontiers(Grid(a,.05), (1.,1.))
        self.assertFalse(goals)
        self.assertEqual(stats['reachable'],0)
    def test_disconnected_room_not_targeted(self):
        a = partial()
        a[:,30:36] = 100
        goals, stats = plan_frontiers(Grid(a,.05),(1.,1.))
        self.assertFalse(goals)
        self.assertEqual(stats['reachable'],0)
        self.assertGreater(stats['disconnected_cells'],0)
    def test_open_door_reachable(self):
        a = partial()
        a[:,30:36] = 100
        a[25:45,30:36] = 0
        goals, stats = plan_frontiers(Grid(a,.05),(1.,1.))
        self.assertTrue(goals)
    def test_too_narrow_door(self):
        a = partial()
        a[:,30:36] = 100
        a[30:35,30:36] = 0
        goals, stats = plan_frontiers(Grid(a,.05),(1.,1.))
        self.assertFalse(goals)
    def test_robot_in_unknown_not_complete(self):
        goals, stats = plan_frontiers(Grid(partial(),.05),(3.,1.))
        self.assertFalse(goals)
        self.assertEqual(stats['reason'],'ROBOT_NOT_IN_SAFE_FREE_SPACE')
    def test_robot_outside_map(self):
        goals, stats = plan_frontiers(Grid(partial(),.05),(-1.,-1.))
        self.assertEqual(stats['reason'],'ROBOT_NOT_IN_SAFE_FREE_SPACE')
    def test_rotated_map(self):
        grid = Grid(partial(), .05, -2, 3, math.pi/2)
        for cell in [(20,20),(0,0),(69,89)]:
            self.assertEqual(grid.cell(*grid.world(*cell)),cell)
        goals, stats = plan_frontiers(grid,grid.world(20,20))
        self.assertTrue(goals)
        self.assertAlmostEqual(goals[0][3],math.pi/2)
    def test_dilation_does_not_wrap(self):
        a = np.zeros((10,10),dtype=bool)
        a[0,0] = True
        b = dilate(a,2)
        self.assertTrue(b[0,1])
        self.assertFalse(b[-1,-1])
    def test_no_diagonal_corner_cutting(self):
        a = np.eye(3,dtype=bool)
        self.assertEqual(flood(a,(0,0))[1,1],-1)
    def test_cooldown_and_retry_limit(self):
        h = GoalHistory()
        h.record((1,1),0,False)
        self.assertFalse(h.allowed((1.1,1),10))
        self.assertTrue(h.allowed((1,1),61))
        h.record((1,1),61,False)
        h.record((1,1),122,False)
        self.assertFalse(h.allowed((1,1),1000))
        self.assertTrue(h.allowed((2,2),1000))
    def test_repeated_success_without_map_gain_bounded(self):
        h = GoalHistory()
        for t in [0,21,42]:
            h.record((1,1),t,True)
        self.assertFalse(h.allowed((1,1),1000))
    def test_nav2_config(self):
        data = {'controller_server':{'ros__parameters':{}},
                'collision_monitor':{'ros__parameters':{'cmd_vel_out_topic':'cmd_vel'}},
                'route_server':{'ros__parameters':{'custom_installed_setting':True}}}
        result = configure(data)
        self.assertTrue(result['controller_server']['ros__parameters']['enable_stamped_cmd_vel'])
        self.assertEqual(result['controller_server']['ros__parameters']['FollowPath']['max_vel_x'],.06)
        self.assertFalse(result['planner_server']['ros__parameters']['GridBased']['allow_unknown'])
        self.assertTrue(result['route_server']['ros__parameters']['custom_installed_setting'])
        self.assertEqual(result['collision_monitor']['ros__parameters']['cmd_vel_out_topic'],'cmd_vel')
    def test_twist_config(self):
        config = configure({'controller_server':{'ros__parameters':{}}},False)
        self.assertFalse(config['velocity_smoother']['ros__parameters']['enable_stamped_cmd_vel'])

if __name__ == '__main__':
    unittest.main()
