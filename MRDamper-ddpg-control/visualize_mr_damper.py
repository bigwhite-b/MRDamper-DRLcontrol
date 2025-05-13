import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from road_generator import *
import config
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置固定的随机种子，确保路面生成的一致性
np.random.seed(42)
# 加载模型参数
def load_model(model_path):
    # 明确指定weights_only=False以解决PyTorch 2.6版本的兼容性问题
    return torch.load(model_path, weights_only=False)

# 磁流变阻尼器模型参数
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

# 初始化磁流变阻尼器状态
mr_state = [0, 0, 0]  # [y, y_dot, z]

# 系统参数
m1 = config.m1  # 车身质量
m2 = config.m2  # 车轮质量
m = config.m    # (m1, m2)
cb = config.cb  # 基础阻尼系数
kb = config.kb  # 基础刚度
kw = config.kw  # 轮胎刚度
dt = config.dt  # 时间步长
TIME = config.TIME  # 时间序列

# 获取路面输入
road_profile = list(RoadProfile().get_profile_by_class("C", config.t_stop, config.dt)[1][1:])

# ODE函数
def odefun(t, y0, m, kw, current=0.0):
    global mr_state
    m1=m[0]    # 车身质量
    m2=m[1]    # 车轮质量

    # 路面条件
    t_idx = min(round(t/dt), len(TIME)-1)
    xr = road_profile[t_idx]

    # 位移和速度
    xb = y0[0]    # 车身位移
    xw = y0[1]    # 车轮位移
    dxb = y0[2]   # 车身速度
    dxw = y0[3]   # 车轮速度
    
    # 计算磁流变阻尼器的力
    y, y_dot, z = mr_state
    
    # 计算c1和alpha
    c1 = cia + cib * current
    alpha = alpha_a + alpha_b * current
    
    # 计算磁流变阻尼器的动态方程
    dydt = (alpha * z + c0 * (dxb-dxw) + k0 * ((xb-xw) - y)) / (c0 + c1)
    dzdt = -gamma * abs((dxb-dxw) - y_dot) * np.sign(z) * (abs(z) ** (n - 1)) \
           - beta * (abs(z) ** n) * np.sign(z) + A * ((dxb-dxw) - y_dot)
    
    # 更新磁流变阻尼器状态
    y_new = y + y_dot * dt
    y_dot_new = y_dot + dydt * dt
    z_new = z + dzdt * dt
    mr_state = [y_new, y_dot_new, z_new]
    
    # 计算磁流变阻尼器产生的力
    mr_force = c1 * y_dot + k1 * ((xb-xw) - y)
    
    # 计算加速度
    d2xb = - kb/m1*(xb-xw) - cb/m1*(dxb-dxw) - mr_force/m1
    d2xw = kb/m2*(xb-xw) + cb/m2*(dxb-dxw) + mr_force/m2 + kw/m2*(xr-xw)

    return [dxb, dxw, d2xb, d2xw]

# 将Actor网络输出映射到电流
def get_current(a):
    return max(0, min(2.0, 1.0 + a))

# 模拟被动悬架系统
def simulate_passive_suspension(road_profile):
    from scipy.integrate import odeint

    # 定义被动悬架系统的动态方程
    def passive_suspension_system(y, t):
        # 状态变量：xb, dxb, xw, dxw
        xb, dxb, xw, dxw = y

        # 获取当前时间步的路面输入
        t_idx = min(int(t / dt), len(road_profile) - 1)
        xr = road_profile[t_idx]

        # 计算被动悬架力
        passive_force = cb * (dxb - dxw) + kb * (xb - xw)

        # 动态方程
        d2xb = -passive_force / m1
        d2xw = passive_force / m2 + kw / m2 * (xr - xw)

        return [dxb, d2xb, dxw, d2xw]

    # 初始条件：[xb, dxb, xw, dxw]
    y0 = [road_profile[0], 0.0, road_profile[0], 0.0]

    # 使用odeint求解微分方程
    solution = odeint(passive_suspension_system, y0, TIME)

    # 直接从结果中提取位移、速度和加速度
    xb_time = solution[:, 0]  # 车身位移
    dxb_time = solution[:, 1]  # 车身速度
    d2xb_time = solution[:, 3]  # 车身加速度 - 直接从odeint结果中提取

    return {
        'xb': xb_time,
        'dxb': dxb_time,
        'd2xb': d2xb_time
    }
# 模拟悬架系统
def simulate_suspension(model, road_profile):
    # 初始化状态
    s_ode = [road_profile[0], road_profile[0], 0, 0]  # [xb, xw, dxb, dxw]
    s_ode_prev = s_ode
    
    # 记录结果
    xb_time = []
    dxb_time = []
    d2xb_time = []
    current_time = []
    mr_force_time = []
    
    # 重置磁流变阻尼器状态
    global mr_state
    mr_state = [0, 0, 0]
    
    # 模拟系统
    for t_idx, t in enumerate(TIME):
        # 获取当前状态
        xb, xw, dxb, dxw = np.array(s_ode)[0:4]
        dxr = (road_profile[t_idx] - road_profile[t_idx-1]) / dt if t_idx > 0 else 0
        xb_prev, xw_prev, dxb_prev, dxw_prev = np.array(s_ode_prev)[0:4]
        dxr_prev = (road_profile[max(t_idx-1, 0)] - road_profile[max(t_idx-2, 0)]) / dt if t_idx > 1 else 0
        
        # 构建RL状态
        s_rl = [dxb, dxw, dxr, dxb_prev, dxw_prev, dxr_prev]
        s_rl = torch.FloatTensor(s_rl).to(config.device)
        
        # 获取动作（电流）
        with torch.no_grad():
            action = model(s_rl).cpu().numpy()
        current = get_current(action[0])
        
        # 计算磁流变阻尼器力
        y, y_dot, z = mr_state
        c1 = cia + cib * current
        mr_force = c1 * y_dot + k1 * ((xb-xw) - y)
        
        # 计算加速度
        d2xb = (-kb/m1*(xb-xw) - cb/m1*(dxb-dxw) - mr_force/m1)
        
        # 记录结果
        xb_time.append(xb)
        dxb_time.append(dxb)
        d2xb_time.append(d2xb)
        current_time.append(current)
        mr_force_time.append(mr_force)
        
        # 求解ODE获取下一状态
        yout = solve_ivp(odefun, [t, t+dt], s_ode, args=(m, kw, current), dense_output=True)
        s_ode_prev = s_ode
        s_ode = yout.y[:,-1]
    
    return {
        'xb': xb_time,
        'dxb': dxb_time,
        'd2xb': d2xb_time,
        'current': current_time,
        'mr_force': mr_force_time,
        'road': road_profile[:len(TIME)]
    }

# 比较不同控制策略
def compare_strategies():
    # 加载训练好的模型
    model_path = r"D:\悬架simulink模型\RL4Suspension-ICMLA23-main\checkpoints\mu_origin-5.12.pt"
    rl_model = load_model(model_path)
    rl_model.eval()
    
    # 生成路面输入
    road_profile = list(RoadProfile().get_profile_by_class("D", config.t_stop, config.dt)[1][1:])
    
    # 模拟RL控制的悬架
    rl_results = simulate_suspension(rl_model, road_profile)
    
    # 模拟被动悬架
    passive_results = simulate_passive_suspension(road_profile)
    
    # 注释掉PID控制部分
    '''
    # 模拟PID控制的磁流变悬架
    global mr_state
    mr_state = [0, 0, 0]
    
    # 创建一个PID控制器模型
    class PIDControlModel:
        def __init__(self):
            self.prev_error = 0
            self.integral = 0
            # PID参数
            self.kp = 105
            self.ki = 0.02
            self.kd = 0.05
            
        def __call__(self, state):
            # 提取状态信息
            dxb = state[0].item()  # 车身速度
            
            # 计算误差（目标是减小车身速度）
            error = -dxb
            
            # 计算PID控制
            self.integral += error * dt
            derivative = (error - self.prev_error) / dt
            self.prev_error = error
            
            # 计算控制输出
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            
            # 限制输出范围在[-1, 1]
            output = max(-1.0, min(1.0, output))
            
            return torch.tensor([output])
    
    pid_results = simulate_suspension(PIDControlModel(), road_profile)
    '''
    
    # 创建一个空的PID结果字典，以保持代码结构一致
    pid_results = {
        'xb': [0] * len(TIME),
        'dxb': [0] * len(TIME),
        'd2xb': [0] * len(TIME),
        'current': [0] * len(TIME),
        'mr_force': [0] * len(TIME)
    }
    
    return {
        'rl': rl_results,
        'pid': pid_results,
        'passive': passive_results,
        'road': road_profile[:len(TIME)]
    }

# 绘制结果
def plot_results(results):
    time = TIME

    # 创建保存路径
    save_dir = r"D:\悬架simulink模型\RL4Suspension-ICMLA23-main\saveimage"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    # 创建一个大图
    plt.figure(figsize=(20, 15))

    # 1. 路面输入
    plt.subplot(5, 1, 1)
    plt.plot(time, results['road'], 'k-')
    plt.title('路面输入')
    plt.ylabel('位移 (m)')
    plt.grid(True)

    # 2. 车身位移
    plt.subplot(5, 1, 2)
    plt.plot(time, results['rl']['xb'], 'r-', label='RL控制')
    # plt.plot(time, results['pid']['xb'], 'g--', label='PID控制')  # 注释掉PID控制
    plt.plot(time, results['passive']['xb'], 'k:', label='被动悬架')
    plt.title('车身位移')
    plt.ylabel('位移 (m)')
    plt.legend()
    plt.grid(True)

    # 3. 车身加速度
    plt.subplot(5, 1, 3)
    plt.plot(time, results['rl']['d2xb'], 'r-', label='RL控制')
    # plt.plot(time, results['pid']['d2xb'], 'g--', label='PID控制')  # 注释掉PID控制
    plt.plot(time, results['passive']['d2xb'], 'k:', label='被动悬架')
    plt.title('车身加速度')
    plt.ylabel('加速度 (m/s^2)')
    plt.legend()
    plt.grid(True)

    # 4. 控制电流
    plt.subplot(5, 1, 4)
    plt.plot(time, results['rl']['current'], 'r-', label='RL控制')
    # plt.plot(time, results['pid']['current'], 'g--', label='PID控制')  # 注释掉PID控制
    plt.plot(time, [0] * len(time), 'k:', label='被动悬架')
    plt.title('控制电流')
    plt.ylabel('电流 (A)')
    plt.legend()
    plt.grid(True)

    # 5. 磁流变阻尼器力
    plt.subplot(5, 1, 5)
    plt.plot(time, results['rl']['mr_force'], 'r-', label='RL控制')
    # plt.plot(time, results['pid']['mr_force'], 'g--', label='PID控制')  # 注释掉PID控制
    plt.plot(time, [cb * (results['passive']['dxb'][i]) + kb * (results['passive']['xb'][i]) for i in range(len(time))], 'k:', label='被动悬架')
    plt.title('阻尼器力')
    plt.ylabel('力 (N)')
    plt.xlabel('时间 (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mr_damper_control_comparison.png'), dpi=300)
    plt.show()

# 计算性能指标
def calculate_metrics(results):
    # 计算RMS值
    def rms(x):
        return np.sqrt(np.mean(np.array(x)**2))
    
    # 计算各种指标
    metrics = {
        'rl': {
            'xb_rms': rms(results['rl']['xb']),
            'd2xb_rms': rms(results['rl']['d2xb']),
            'current_mean': np.mean(results['rl']['current']),
            'mr_force_rms': rms(results['rl']['mr_force'])
        },
        # 注释掉PID控制的指标
        'pid': {
            'xb_rms': 0,
            'd2xb_rms': 0,
            'current_mean': 0,
            'mr_force_rms': 0
        },
        'passive': {
            'xb_rms': rms(results['passive']['xb']),
            'd2xb_rms': rms(results['passive']['d2xb']),
            'current_mean': 0,
            'mr_force_rms': rms([cb * (results['passive']['dxb'][i]) + kb * (results['passive']['xb'][i]) for i in range(len(TIME))])
        }
    }
    
    # 打印指标
    print("性能指标比较：")
    print("-" * 60)
    print(f"{'指标':<15} {'RL控制':<15} {'被动悬架':<15}")  # 移除PID控制列
    print("-" * 60)
    print(f"{'车身位移RMS':<15} {metrics['rl']['xb_rms']:<15.6f} {metrics['passive']['xb_rms']:<15.6f}")
    print(f"{'车身加速度RMS':<15} {metrics['rl']['d2xb_rms']:<15.6f} {metrics['passive']['d2xb_rms']:<15.6f}")
    print(f"{'平均电流':<15} {metrics['rl']['current_mean']:<15.6f} {metrics['passive']['current_mean']:<15.6f}")
    print(f"{'阻尼力RMS':<15} {metrics['rl']['mr_force_rms']:<15.6f} {metrics['passive']['mr_force_rms']:<15.6f}")
    print("-" * 60)
    
    return metrics


# # 主函数
if __name__ == "__main__":
    print("开始可视化磁流变阻尼器控制的悬架系统...")

    # 创建保存路径
    save_dir = r"D:\悬架simulink模型\RL4Suspension-ICMLA23-main\saveimage"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    # 比较不同控制策略
    results = compare_strategies()

    # 绘制结果
    plot_results(results)

    # 计算性能指标
    metrics = calculate_metrics(results)

    # 计算峰值加速度
    peak_acc_rl = np.max(np.abs(results['rl']['d2xb']))
    # peak_acc_pid = np.max(np.abs(results['pid']['d2xb']))  # 注释掉PID控制
    peak_acc_passive = np.max(np.abs(results['passive']['d2xb']))

    print("\n峰值加速度比较：")
    print("-" * 60)
    print(f"{'控制策略':<15} {'峰值加速度':<15} {'相对于被动悬架':<15}")
    print("-" * 60)
    print(f"{'RL控制':<15} {peak_acc_rl:<15.6f} {(peak_acc_rl - peak_acc_passive) / peak_acc_passive * 100:+.2f}%")
    # print(f"{'PID控制':<15} {peak_acc_pid:<15.6f} {(peak_acc_pid - peak_acc_passive) / peak_acc_passive * 100:+.2f}%")  # 注释掉PID控制
    print(f"{'被动悬架':<15} {peak_acc_passive:<15.6f} {0:+.2f}%")
    print("-" * 60)

    print("可视化完成！")


