#!/usr/bin/env python3
"""Adapt the INSTALLED Nav2 defaults, preserving version-specific server config.
Writes a new local YAML. Does not modify installed ROS packages.
"""
import argparse
from pathlib import Path
import yaml


def configure(data, stamped=True):
    def params(name):
        return data.setdefault(name, {}).setdefault('ros__parameters', {})

    def clocks(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'ros__parameters' and isinstance(value, dict):
                    value['use_sim_time'] = True
                clocks(value)
    clocks(data)
    for name in ('controller_server', 'behavior_server', 'velocity_smoother',
                 'collision_monitor', 'bt_navigator', 'docking_server'):
        if name in data:
            params(name)['enable_stamped_cmd_vel'] = stamped
    bt = params('bt_navigator')
    bt.update(global_frame='map', robot_base_frame='base_footprint', odom_topic='/odom')
    controller = params('controller_server')
    controller.update(controller_frequency=10., min_x_velocity_threshold=.001,
                      min_y_velocity_threshold=.001, min_theta_velocity_threshold=.001,
                      controller_plugins=['FollowPath'], progress_checker_plugins=['progress_checker'],
                      goal_checker_plugins=['general_goal_checker'], enable_stamped_cmd_vel=stamped)
    controller['progress_checker'] = dict(plugin='nav2_controller::SimpleProgressChecker',
                                         required_movement_radius=.10, movement_time_allowance=20.)
    controller['general_goal_checker'] = dict(plugin='nav2_controller::SimpleGoalChecker',
                                             stateful=True, xy_goal_tolerance=.08, yaw_goal_tolerance=.25)
    controller['FollowPath'] = {
        'plugin': 'dwb_core::DWBLocalPlanner', 'debug_trajectory_details': False,
        'min_vel_x': 0., 'min_vel_y': 0., 'max_vel_x': .06, 'max_vel_y': 0.,
        'max_vel_theta': .4, 'min_speed_xy': 0., 'max_speed_xy': .06,
        'min_speed_theta': 0., 'acc_lim_x': .25, 'acc_lim_y': 0., 'acc_lim_theta': .8,
        'decel_lim_x': -.25, 'decel_lim_y': 0., 'decel_lim_theta': -.8,
        'vx_samples': 12, 'vy_samples': 1, 'vtheta_samples': 20, 'sim_time': 2.,
        'linear_granularity': .025, 'angular_granularity': .025,
        'transform_tolerance': .2, 'xy_goal_tolerance': .08,
        'trans_stopped_velocity': .01, 'short_circuit_trajectory_evaluation': True,
        'stateful': True,
        'critics': ['RotateToGoal', 'Oscillation', 'BaseObstacle', 'GoalAlign', 'PathAlign', 'PathDist', 'GoalDist'],
        'BaseObstacle.scale': .02, 'PathAlign.scale': 24., 'PathAlign.forward_point_distance': .1,
        'GoalAlign.scale': 16., 'GoalAlign.forward_point_distance': .1,
        'PathDist.scale': 24., 'GoalDist.scale': 16., 'RotateToGoal.scale': 24.,
        'RotateToGoal.slowing_factor': 5., 'RotateToGoal.lookahead_time': -1.,
    }
    for name in ('local_costmap', 'global_costmap'):
        p = data.setdefault(name, {}).setdefault(name, {}).setdefault('ros__parameters', {})
        p.update(use_sim_time=True, robot_base_frame='base_footprint', robot_radius=.15,
                 footprint_padding=.01, resolution=.05, always_send_full_costmap=True)
        p.pop('footprint', None)
        if name == 'global_costmap':
            p.update(global_frame='map', track_unknown_space=True,
                     plugins=['static_layer', 'obstacle_layer', 'inflation_layer'])
            p['static_layer'] = dict(plugin='nav2_costmap_2d::StaticLayer', map_topic='/map',
                                    map_subscribe_transient_local=False, subscribe_to_updates=True)
        else:
            p.update(global_frame='odom', rolling_window=True, width=3, height=3,
                     plugins=['obstacle_layer', 'inflation_layer'])
        p['obstacle_layer'] = {
            'plugin': 'nav2_costmap_2d::ObstacleLayer', 'enabled': True, 'observation_sources': 'scan',
            'scan': {'topic': '/scan', 'data_type': 'LaserScan', 'clearing': True, 'marking': True,
                     'inf_is_valid': True, 'obstacle_max_range': 3., 'obstacle_min_range': 0.,
                     'raytrace_max_range': 3.5, 'raytrace_min_range': 0., 'max_obstacle_height': 2.}}
        p['inflation_layer'] = dict(plugin='nav2_costmap_2d::InflationLayer',
                                   inflation_radius=.35, cost_scaling_factor=5.)
    planner = params('planner_server')
    planner['planner_plugins'] = ['GridBased']
    planner['GridBased'] = dict(plugin='nav2_navfn_planner::NavfnPlanner', tolerance=.10,
                               use_astar=True, allow_unknown=False)
    behavior = params('behavior_server')
    behavior.update(robot_base_frame='base_footprint', local_frame='odom', global_frame='map',
                    max_rotational_vel=.4, min_rotational_vel=.1, rotational_acc_lim=.8,
                    enable_stamped_cmd_vel=stamped)
    smoother = params('velocity_smoother')
    smoother.update(max_velocity=[.06, 0., .4], min_velocity=[-.04, 0., -.4],
                    max_accel=[.25, 0., .8], max_decel=[-.25, 0., -.8],
                    velocity_timeout=.5, odom_topic='/odom', enable_stamped_cmd_vel=stamped)
    if 'collision_monitor' in data:
        params('collision_monitor').update(base_frame_id='base_footprint', enable_stamped_cmd_vel=stamped)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Optional installed Nav2 defaults YAML')
    parser.add_argument('--output', default='nav2_exploration.yaml')
    parser.add_argument('--twist', action='store_true', help='Use unstamped Twist, only if the base requires it')
    args = parser.parse_args()
    if args.source:
        source = Path(args.source)
    else:
        from ament_index_python.packages import get_package_share_directory
        source = Path(get_package_share_directory('nav2_bringup'))/'params'/'nav2_params.yaml'
    destination = Path(args.output)
    if destination.resolve() == source.resolve():
        raise SystemExit('Refusing to overwrite installed source YAML')
    with source.open() as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or 'controller_server' not in data:
        raise SystemExit('Source is not a full Nav2 parameter file')
    destination.write_text(yaml.safe_dump(configure(data, not args.twist), sort_keys=False))
    print(f'Created {destination.resolve()} from {source}')
    print('Command type: ' + ('Twist' if args.twist else 'TwistStamped'))


if __name__ == '__main__':
    main()
