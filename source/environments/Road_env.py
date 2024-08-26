import time
import numpy as np  # but trying to avoid using it (np.array cannot be converted to JSON)
import tkinter as tk
from source.utils.logger import Logger

UNIT = 20   # pixels per grid cell
MAZE_H = 2  # grid height
MAZE_W = 20  # grid width  !! Adapt the threshold_success in the main accordingly
HALF_SIZE = UNIT * 0.35  # half size factor of square in a cell
Y_COORD = 0  # blocking motion of ego agent in one row - no vertical motion allowed

def one_hot_encoding(feature_to_encode):
    repr_size = MAZE_W
    one_hot_list = np.zeros(repr_size)
    # one_hot_list = [0] * UNIT
    if feature_to_encode < repr_size:
        one_hot_list[feature_to_encode] = 1
    else:
        print('feature is out of scope for one_hot_encoding: %i / %i' % (feature_to_encode, repr_size))
    # print(one_hot_list)
    return one_hot_list

def process_state(input_state):
    ego_position = input_state[0]
    velocity = input_state[1]
    return [ego_position, velocity]  # , obstacle_position]

class Road(tk.Tk, object):  # 定义了 Road 类，用于模拟道路环境
    def __init__(self, using_tkinter, actions_names, state_features, initial_state, goal_velocity=4):
        """
        :param using_tkinter: [bool] 图形界面标志
        :param actions_names: [string] 可能的动作列表
        :param state_features: [string] 构成状态的特征列表。字符串！！
        :param initial_state: [list of int] 初始状态
        :param goal_velocity: [int] 目标速度
        """
        # 图形界面
        if using_tkinter:
            super(Road, self).__init__()
        # 动作空间
        self.actions_list = actions_names
        # 状态由以下部分组成
        # - 绝对自车位置
        # - 速度
        # - 障碍物的绝对位置
        self.state_features = state_features
        # 奖励 - 奖励在以下过程中更新
        # - 在转换期间（硬约束）
        # - 在reward_function中（仅考虑新状态）
        self.reward = 0
        self.rewards_dict = {
            # 效率 = 进展
            "goal_with_good_velocity": 40,#当车辆以目标速度到达终点时的奖励。
            "goal_with_bad_velocity": -40,#当车辆未以目标速度到达终点时的惩罚。
            "per_step_cost": -3,#每一步的成本，防止不必要的操作。
            "under_speed": -15,#低于最小速度的惩罚。
            # 交通法规
            "over_speed": -10,#超速的惩罚。
            "over_speed_2": -10,
            # 安全
            "over_speed_near_pedestrian": -40,# 在靠近行人时超速的惩罚。
            # 舒适性
            "negative_speed": -15,
            "action_change": -2 # 动作变化的惩罚。
        }
        # 定义速度限制
        self.max_velocity_1 = 4
        self.max_velocity_2 = 2 #这个参数在文件中并未被显式使用。如果需要，可以在特定条件下（如在特定路段或靠近某些障碍物时）设置更严格的速度限制。
        self.max_velocity_pedestrian = 2
        self.min_velocity = 0
        # 状态 - 暂时为不同的变量
        self.initial_state = initial_state
        self.state_ego_position = self.initial_state[0]
        self.state_ego_velocity = self.initial_state[1]
        self.state_obstacle_position = self.initial_state[2]
        self.previous_state_position = self.state_ego_position
        self.previous_state_velocity = self.state_ego_velocity
        self.previous_action = None
        # 环境：这里调整行人的位置------------------
        self.goal_coord = [MAZE_W - 1, 0]
        self.goal_velocity = goal_velocity
        self.obstacle1_coord = [self.state_obstacle_position, 1]
        #self.obstacle2_coord = [1, 3]
        self.initial_position = [self.initial_state[0], Y_COORD]
        # 根据速度调整代理的颜色
        colours_list = ["white", "yellow", "orange", "red2", "red3", "red4", "black", "black", "black", "black",
                        "black", "black"]
        velocity_list = range(len(colours_list) + 1)
        self.colour_velocity_code = dict(zip(velocity_list, colours_list))
        # 图形界面
        self.using_tkinter = using_tkinter
        # 在Tk框架中创建原点
        self.origin_coord = [(x + y) for x, y in zip(self.initial_position, [0.5, 0.5])]
        self.origin = [x * UNIT for x in self.origin_coord]
        self.canvas = None
        self.rect = None
        self.obstacle = None
        if self.using_tkinter:
            # Tk窗口
            self.title('road')
            self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
            self.build_road()
        # 日志配置
        self.logger = Logger("road", "road_env.log", 0)

    @staticmethod
    def sample_position_obstacle():
        # 固定障碍物位置为12
        fix_position_obstacle = 12
        return fix_position_obstacle

    def build_road(self):
        """
        构建Tk窗口
        仅在开始时调用一次
        :return:  无
        """
        if self.using_tkinter:
            # 创建画布
            self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

            # 创建网格
            for c in range(0, MAZE_W * UNIT, UNIT):
                x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
                self.canvas.create_line(x0, y0, x1, y1)
            for r in range(0, MAZE_H * UNIT, UNIT):
                x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
                self.canvas.create_line(x0, y0, x1, y1)

            # 创建自车代理
            self.rect = self.canvas.create_rectangle(
                self.origin[0] - HALF_SIZE, self.origin[1] - HALF_SIZE,
                self.origin[0] + HALF_SIZE, self.origin[1] + HALF_SIZE,
                fill='white')

            # 障碍物1
            obstacle1_center = np.asarray(self.origin) + UNIT * np.asarray(self.obstacle1_coord)
            self.obstacle = self.canvas.create_rectangle(
                obstacle1_center[0] - HALF_SIZE, obstacle1_center[1] - HALF_SIZE,
                obstacle1_center[0] + HALF_SIZE, obstacle1_center[1] + HALF_SIZE,
                fill='black')
            '''
            # 障碍物2
            obstacle2_center = np.asarray(self.origin) + UNIT * np.asarray(self.obstacle2_coord)
            self.canvas.create_rectangle(
                obstacle2_center[0] - HALF_SIZE, obstacle2_center[1] - HALF_SIZE,
                obstacle2_center[0] + HALF_SIZE, obstacle2_center[1] + HALF_SIZE,
                fill='black')
            '''
            # 创建目标的椭圆
            goal_center = np.asarray(self.origin) + UNIT * np.asarray(self.goal_coord)
            self.canvas.create_oval(
                goal_center[0] - HALF_SIZE, goal_center[1] - HALF_SIZE,
                goal_center[0] + HALF_SIZE, goal_center[1] + HALF_SIZE,
                fill='yellow')


            # 打包所有组件
            self.canvas.pack()

    def reset(self):
        """
        清除画布（移除代理）
        清除状态（重新初始化）
        为障碍物采样一个随机位置
        :return: 初始状态和被屏蔽的动作列表
        """
        time.sleep(0.005)
        random_position_obstacle = self.sample_position_obstacle()
        self.obstacle1_coord = [random_position_obstacle, 1]#重置障碍物

        if self.using_tkinter:
            self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(
                self.origin[0] - HALF_SIZE, self.origin[1] - HALF_SIZE,
                self.origin[0] + HALF_SIZE, self.origin[1] + HALF_SIZE,
                fill='white')

            self.canvas.delete(self.obstacle)
            obstacle1_center = np.asarray(self.origin) + UNIT * np.asarray(self.obstacle1_coord)
            self.obstacle = self.canvas.create_rectangle(
                obstacle1_center[0] - HALF_SIZE, obstacle1_center[1] - HALF_SIZE,
                obstacle1_center[0] + HALF_SIZE, obstacle1_center[1] + HALF_SIZE,
                fill='black')

        self.state_ego_position = self.initial_state[0]
        self.state_ego_velocity = self.initial_state[1]
        self.state_obstacle_position = random_position_obstacle
        self.initial_state[2] = random_position_obstacle
        self.previous_state_position = self.initial_state[0]
        self.previous_state_velocity = self.initial_state[1]

        return process_state(self.initial_state), self.masking_function()

    def transition(self, action, velocity=None):
        """
        根据所需指令更新状态中的速度
        :param action: 所需动作
        :param velocity: [可选]
        :return: 更新后的速度
        """
        if velocity is None:
            velocity = self.state_ego_velocity

        delta_velocity = 0

        if action == self.actions_list[0]:  # 保持速度
            delta_velocity = 0
        elif action == self.actions_list[1]:  # 加速
            delta_velocity = 1
        elif action == self.actions_list[2]:  # 大幅加速
            delta_velocity = 2
        elif action == self.actions_list[3]:  # 减速
            delta_velocity = -1
        elif action == self.actions_list[4]:  # 大幅减速
            delta_velocity = -2

        return velocity + delta_velocity # vx(t+1) = vx(t) + a(t)∆t

    def step(self, action):
        """
        将动作转换为新状态
        - 调用转移模型
        - 调用检查硬条件
        - 调用屏蔽 - 待实现

        :param action: [string] 所需动作
        :return: 元组，包含：
        - 新状态（列表）
        - 奖励（int）
        - 终止标志（bool）
        - 被屏蔽的动作列表（list）
        """
        self.previous_state_velocity = self.state_ego_velocity
        self.previous_state_position = self.state_ego_position
        self.previous_action = action

        # 转移 = 获取新状态：
        self.state_ego_velocity = self.transition(action)
        if self.state_ego_velocity < 0:
            self.state_ego_velocity = 0
            message = "self.state_ego_velocity不能小于0 - a = {} - p = {} - v = {} 在step()".format(action, self.state_ego_position, self.state_ego_velocity)
            self.logger.log(message, 3)

        # 假设简单关系：速度以[step/sec]表示，时间步长为1秒
        desired_position_change = self.state_ego_velocity

        # 将速度信息转换为位置变化 = 步长数
        tk_update_steps = [0, 0]

        # 更新状态 - 位置
        self.state_ego_position = self.state_ego_position + desired_position_change
        tk_update_steps[0] += desired_position_change * UNIT

        if self.using_tkinter:
            # 在画布上移动代理
            self.canvas.move(self.rect, tk_update_steps[0], tk_update_steps[1])
            # 根据速度更新颜色
            new_colour = self.colour_velocity_code[self.state_ego_velocity]
            self.canvas.itemconfig(self.rect, fill=new_colour)

        # 观察奖励
        [reward, termination_flag] = self.reward_function(action)

        # 对于下一次决策，这些动作不可用（它使用输出状态）：
        if termination_flag:
            masked_actions_list = []
        else:
            masked_actions_list = self.masking_function()

        state_to_return = process_state([self.state_ego_position, self.state_ego_velocity, self.state_obstacle_position])
        return state_to_return, reward, termination_flag, masked_actions_list

    def reward_function(self, action):
        """
        ToDo: 规范化它
        待优化
        - 需要考虑从以前状态到新状态的所有中间点
        :return: 奖励（int）和终止标志（bool）
        """
        self.reward = 0

        # 惩罚动作变化
        if self.state_ego_velocity != self.previous_state_velocity:
            change_in_velocity = self.state_ego_velocity - self.previous_state_velocity
            self.reward += self.rewards_dict["action_change"] * abs(change_in_velocity)

        # 目标位置
        if self.state_ego_position >= self.goal_coord[0]:
            self.state_ego_position = self.goal_coord[0]
            if self.state_ego_velocity == self.goal_velocity:
                self.reward += self.rewards_dict["goal_with_good_velocity"]
            else:
                self.reward += self.rewards_dict["goal_with_bad_velocity"]
            termination_flag = True #抵达目标位置

        else:
            self.reward += self.rewards_dict["per_step_cost"]
            termination_flag = False

        # 检查最大速度限制
        if self.state_ego_velocity > self.max_velocity_1:
            excess_in_velocity = self.state_ego_velocity - self.max_velocity_1
            self.reward += self.rewards_dict["over_speed"] * excess_in_velocity
            message = "Too fast! in reward_function() -- hard constraints should have masked it. " \
                      "a = {} - p = {} - v = {}".format(action, self.state_ego_position, self.state_ego_velocity)
            self.logger.log(message, 3)

        # 检查最低速度
        if self.state_ego_velocity < self.min_velocity:
            excess_in_velocity = self.min_velocity - self.state_ego_velocity
            self.reward += self.rewards_dict["under_speed"] * excess_in_velocity

            if self.state_ego_velocity < 0:
                excess_in_velocity = abs(self.min_velocity)
                message = "Under speed! in reward_function() -- hard constraints should have masked it. " \
                          "a = {} - p = {} - v = {}".format(action, self.state_ego_position, self.state_ego_velocity)
                self.logger.log(message, 3)

                self.state_ego_velocity = 0
                self.reward += self.rewards_dict["negative_speed"] * excess_in_velocity

        # 限制靠近行人时的速度
        if self.previous_state_position <= self.obstacle1_coord[0] <= self.state_ego_position:
            if self.state_ego_velocity > self.max_velocity_pedestrian:
                excess_in_velocity = self.state_ego_velocity - self.max_velocity_pedestrian
                self.reward += self.rewards_dict["over_speed_near_pedestrian"] * excess_in_velocity
                message = "Too fast close to obstacle! in reward_function() - a = {} - p = {} - po= {} - v = {}".format(action, self.state_ego_position, self.state_obstacle_position, self.state_ego_velocity)
                self.logger.log(message, 1)

        return self.reward, termination_flag

    def masking_function(self, state=None):
        """
        硬约束
        使用状态（位置，速度）

        :param state: [可选]
        :return: 被屏蔽的动作列表（从self.action_list中获取的子列表）
        """
        if state is None:
            velocity = None
        else:
            velocity = state[1]

        masked_actions_list = []

        # 检查是否达到最大/最小速度
        for action_candidate in self.actions_list:
            velocity_candidate = self.transition(action_candidate, velocity)
            if velocity_candidate > self.max_velocity_1:
                masked_actions_list.append(action_candidate)
            elif velocity_candidate < 0:
                masked_actions_list.append(action_candidate)

        # 检查是否还有可能的动作：
        if masked_actions_list == self.actions_list:
            print("警告 - 速度 %s 和 位置 %s" % (self.state_ego_velocity, self.state_ego_position))
            message = "在masking_function()中找不到可能的动作！ a = {} - p = {} - po= {} - v = {}".format(self.previous_action, self.state_ego_position, self.state_obstacle_position, self.state_ego_velocity)
            self.logger.log(message, 4)

        return masked_actions_list

    def move_to_state(self, state):
        """
        用于基于模型的DP的传送，动态规划，需要网络
        :return: 无
        """
        self.state_ego_position = state[0]
        self.state_ego_velocity = state[1]

    def render(self, sleep_time):#显示代理在环境中的当前状态，主要用于演示和可视化
        """
        :param sleep_time: [float]
        用于demo()
        :return: 无
        """
        if self.using_tkinter:
            time.sleep(sleep_time)
            self.update()

def demo(actions, nb_episodes_demo):
    for t in range(nb_episodes_demo):
        _, masked_actions_list = env.reset()
        print(f"Episode {t + 1}")
        while True:
            env.render(0.5) if env.using_tkinter else None
            possible_actions = [action for action in actions if action not in masked_actions_list]
            if not possible_actions:
                print("WARNING - No possible actions!")
            action = np.random.choice(possible_actions)
            state, reward, termination_flag, masked_actions_list = env.step(action)
            print(f"Action = {action}, State = {state}, Reward = {reward}, Termination flag = {termination_flag}")
            if termination_flag:
                break

if __name__ == '__main__':
    flag_tkinter = True
    actions_list = ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
    the_initial_state = [0, 3, 12]
    env = Road(flag_tkinter, actions_list, ["position", "velocity"], the_initial_state)
    env.after(100, demo, actions_list, 5) if flag_tkinter else demo(actions_list, 5)
    env.mainloop() if flag_tkinter else None


