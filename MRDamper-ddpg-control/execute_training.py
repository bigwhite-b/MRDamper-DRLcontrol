import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from scipy.integrate import solve_ivp
from tqdm.notebook import tqdm
from road_generator import *
import config
from model import *
from trainer import *

np.random.seed(config.seed)
torch.manual_seed(config.seed)

""" ODE Systems parameters """
m1 = config.m1
m2 = config.m2 
m = config.m
cb = config.cb  # 固定的基础阻尼系数
kb = config.kb  # 固定的基础刚度
kw = config.kw
dt = config.dt
TIME = config.TIME

""" 磁流变阻尼器模型参数 """
k0 = 10     # 刚度 (k0)
k1 = 135    # 刚度 (k1)
c0 = 1155   # 粘滞阻尼系数 (c0)
cia = 8185  # c1的常数项 (cia)
cib = 2750  # c1的电流系数 (cib)
alpha_a = 0    # α的常数项 (αa)
alpha_b = 1745 # α的电流系数 (αb)
gamma = 0.0008 # Bouc-Wen模型参数 γ
beta = 0.0001  # Bouc-Wen模型参数 β
A = 350        # Bouc-Wen模型参数 A
n = 2          # Bouc-Wen模型的非线性指数

"""**********************************************************
    MR Damper Environment Class
    - Encapsulates the MR damper state
    - Provides reset and step methods
**********************************************************"""
class MRDamperEnv:
    def __init__(self):
        self.mr_state = [0, 0, 0]  # [y, y_dot, z]
        self.s_ode = [0, 0, 0, 0]  # [xb, xw, dxb, dxw]
        self.s_ode_prev = [0, 0, 0, 0]
        self.road_profile = None
        self.reset()
    
    def reset(self, road_profile=None):
        # Reset MR damper state
        self.mr_state = [0, 0, 0]
        
        # Set road profile
        if road_profile is not None:
            self.road_profile = road_profile
        else:
            self.road_profile = list(RoadProfile().get_profile_by_class("E", config.t_stop, config.dt)[1][1:])
        
        # Initialize state
        self.s_ode = [self.road_profile[0], self.road_profile[0], 0, 0]
        self.s_ode_prev = self.s_ode.copy()
        
        # Return initial observation
        return self._get_observation(0)
    
    def _get_observation(self, t_idx):
        xb, xw, dxb, dxw = self.s_ode
        dxr = (self.road_profile[t_idx] - self.road_profile[t_idx-1]) / dt if t_idx > 0 else 0
        xb_prev, xw_prev, dxb_prev, dxw_prev = self.s_ode_prev
        dxr_prev = (self.road_profile[max(t_idx-1, 0)] - self.road_profile[max(t_idx-2, 0)]) / dt if t_idx > 1 else 0
        
        return [dxb, dxw, dxr, dxb_prev, dxw_prev, dxr_prev]
    
    def step(self, action, t, t_idx):
        # Get current state
        xb, xw, dxb, dxw = self.s_ode
        
        # Get current from action
        current = get_current(action[0])
        
        # 使用欧拉法代替solve_ivp
        # 1. 计算状态导数
        dydt = self._dynamics(self.s_ode, current, t_idx)
        
        # 2. 使用欧拉法更新状态
        self.s_ode_prev = self.s_ode.copy()
        self.s_ode = [self.s_ode[i] + dt * dydt[i] for i in range(4)]
        
        # 3. 裁剪状态（防止数值爆炸）
        self.s_ode = [np.clip(val, -1e4, 1e4) for val in self.s_ode]
        
        # 获取更新后的状态
        xb_next, xw_next, dxb_next, dxw_next = self.s_ode
        
        # 获取更新后的MR阻尼器状态
        y_next, y_dot_next, z_next = self.mr_state
        
        # 计算MR阻尼器力 - 使用更新后的状态
        c1 = cia + cib * current
        mr_force = c1 * y_dot_next + k1 * ((xb_next-xw_next) - y_next)

        # 计算加速度 - 使用更新后的状态和力
        d2xb = (-kb/m1*(xb_next-xw_next) - cb/m1*(dxb_next-dxw_next) - mr_force/m1)
        
        # Get next observation
        next_t_idx = min(t_idx + 1, len(TIME) - 1)
        next_obs = self._get_observation(next_t_idx)
        
        # Reward function
        reward = - 1/10 * abs(d2xb)
        
        # Check if done
        done = (t_idx >= len(TIME) - 1)
        
        # Additional info
        info = {
            'xb': xb_next,
            'dxb': dxb_next,
            'd2xb': d2xb,
            'current': current,
            'mr_force': mr_force
        }
        
        return next_obs, reward, done, info
    
    def _dynamics(self, state, current, t_idx):
        """使用欧拉法计算状态导数
        
        Args:
            state: 当前状态 [xb, xw, dxb, dxw]
            current: 控制电流
            t_idx: 时间索引
            
        Returns:
            状态导数 [dxb, dxw, d2xb, d2xw]
        """
        # 提取状态变量
        xb, xw, dxb, dxw = state
        
        # 获取路面输入
        xr = self.road_profile[min(t_idx, len(self.road_profile)-1)]
        
        # 更新MR阻尼器状态
        y, y_dot, z = self.mr_state
        
        # 计算c1和alpha
        c1 = cia + cib * current
        alpha = alpha_a + alpha_b * current
        
        # 计算MR阻尼器动态方程
        dydt = (alpha * z + c0 * (dxb-dxw) + k0 * ((xb-xw) - y)) / (c0 + c1)
        dzdt = -gamma * abs((dxb-dxw) - y_dot) * np.sign(z) * (abs(z) ** (n - 1)) \
               - beta * (abs(z) ** n) * np.sign(z) + A * ((dxb-dxw) - y_dot)
        
        # 使用欧拉法更新MR阻尼器状态
        y_new = y + dt * y_dot
        y_dot_new = y_dot + dt * dydt
        z_new = z + dt * dzdt
        
        # 裁剪MR阻尼器状态（防止数值爆炸）
        y_new = np.clip(y_new, -1e4, 1e4)
        y_dot_new = np.clip(y_dot_new, -1e4, 1e4)
        z_new = np.clip(z_new, -1e4, 1e4)
        
        self.mr_state = [y_new, y_dot_new, z_new]
        
        # 计算MR阻尼器力 - 使用更新后的状态
        mr_force = c1 * y_dot_new + k1 * ((xb-xw) - y_new)
        
        # 计算加速度
        d2xb = - kb/m1*(xb-xw) - cb/m1*(dxb-dxw) - mr_force/m1
        d2xw = kb/m2*(xb-xw) + cb/m2*(dxb-dxw) + mr_force/m2 + kw/m2*(xr-xw)
        
        return [dxb, dxw, d2xb, d2xw]

"""**********************************************************
    Map Actor Network's output to MR damper current
    Range: [0, 2.0] A
**********************************************************"""
def get_current(a):
    # 将输出映射到0-2.0安培范围
    return max(0, min(2.0, 1.0 + a))

"""**********************************************************
    Execute training process
**********************************************************"""
def execute_training(ckpt_Q_origin, ckpt_mu_origin):
    """ Initialize Trainer (model weights & optimizers)"""
    trainer = Trainer(ckpt_Q_origin=ckpt_Q_origin, 
                      ckpt_mu_origin=ckpt_mu_origin)
                      
    """ Replay buffer """
    buffer = replayBuffer(buffer_size=100000)

    """ Training history"""
    reward_records = []
    best_score = -99999
    rl_current = []  # 记录电流值
    
    """ Initialize environment """
    env = MRDamperEnv()

    """ Episode Iteration"""
    for episode in tqdm(range(config.num_episodes)):
        """--------------------------------------------------------------
            1. Randomly Generate new road profile each episode
        --------------------------------------------------------------"""
        road_profile = list(RoadProfile().get_profile_by_class("E", config.t_stop, config.dt)[1][1:])
        
        """--------------------------------------------------------------
                Exploration factor 
        --------------------------------------------------------------"""
        output_dim = 1  # 现在只有一个输出：电流
        if episode < 200:  # 增加高探索阶段的时间
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.6)  # 增加初始探索强度
        elif episode < 400:  # 增加中等探索阶段的时间
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.4)  # 增加中期探索强
        else:
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.2)  # 增加后期探索强度

        """--------------------------------------------------------------
            2. Reset environment and initialize variables
        --------------------------------------------------------------"""
        s_rl = env.reset(road_profile)
        done = False
        total_reward = 0
        temp_xb_time = []
        temp_dxb_time = []
        
        """--------------------------------------------------------------
                3. Each Episode optimization
        --------------------------------------------------------------"""
        for t_idx, t in enumerate(config.TIME):
            """####################################
                I. State & Action: S{t} & A{t} COMPUTATION
                    (current State & Action)
            ####################################"""
            """ 1.1. Get action from current state """
            a = trainer.pick_sample(s_rl, ou_action_noise)
            
            """ 1.2. Take step in environment """
            s_rl_next, r, done, info = env.step(a, t, t_idx)
            
            """ 1.3. Record data """
            temp_xb_time.append(info['xb'])
            temp_dxb_time.append(info['dxb'])
            rl_current.append(info['current'])
            total_reward += r
            
            """####################################
                II. BUFFER & MODEL UPDATES
            ####################################"""
            """ 2.1. Get next action with no noise """
            ou_action_noise_zero = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.0)
            a_next = trainer.pick_sample(s_rl_next, ou_action_noise_zero)
            
            """ 2.2. Buffer experience """
            buffer.add([s_rl, a, r, s_rl_next, float(done)])
            
            """ 2.3. Update model based on buffered experience """
            if buffer.length() >= config.bs:
                states, actions, rewards, n_states, dones = buffer.sample(config.bs)
                trainer.optimize(states, actions, rewards, n_states, dones)
                trainer.update_target()

            """####################################
                III. STATE TRANSITION
            ####################################"""
            s_rl = s_rl_next
            
            if done:
                break

        # Output total rewards in episode
        print(f"Run episode {episode} with rewards {total_reward}")
        reward_records.append(total_reward)
        np.save('Reward.npy', np.array(reward_records))

        if best_score < total_reward:
            best_score = total_reward
            print(f'New best score: {best_score}')
            rl_xb_time = temp_xb_time
            rl_dxb_time = temp_dxb_time
            trainer.save_checkpoints()

# Keep the replay buffer and noise classes unchanged
"""**********************************************************
    Replay buffer class to enrich past experience
    ->  avoid biased toward recent experience
        (hence forgot past interactions / rare situations)
**********************************************************"""
class replayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        if self._next_idx == self.buffer_size - 1:
            self._next_idx = 0
        else:
            self._next_idx = self._next_idx + 1

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        states   = [self.buffer[i][0] for i in indices]
        actions  = [self.buffer[i][1] for i in indices]
        rewards  = [self.buffer[i][2] for i in indices]
        n_states = [self.buffer[i][3] for i in indices]
        dones    = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)

"""**********************************************************
    Ornstein-Uhlenbeck noise implemented by OpenAI
    Copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
**********************************************************"""
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

if __name__ == '__main__':
    save_dir = r'D:\悬架simulink模型\RL4Suspension-ICMLA23-main\checkpoints'
    ckpt_Q_origin = f'{save_dir}/Q_origin-5.13.pt'
    ckpt_mu_origin = f'{save_dir}/mu_origin-5.13.pt'
    print('Start Training Process...')

    execute_training(ckpt_Q_origin, ckpt_mu_origin)
