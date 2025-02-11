 import numpy as np
import matplotlib.pyplot as plt
# 添加这一行来设置后端
import matplotlib
matplotlib.use('Agg')

def plot_heart_curve():
    """绘制心形曲线"""
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 方法1：参数方程实现
    def heart_curve1():
        t = np.linspace(0, 2*np.pi, 1000)
        x = 16 * np.sin(t)**3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        plt.plot(x, y, 'r-', label='Parametric Equation')
    
    # 方法2：笛卡尔心形曲线
    def heart_curve2():
        t = np.linspace(-2, 2, 1000)
        x = np.linspace(-2, 2, 1000)
        X, Y = np.meshgrid(x, t)
        Z = (X**2 + Y**2 - 1)**3 - X**2 * Y**3
        plt.contour(X, Y, Z, [0], colors='pink')  # 移除 label 参数
    
    # 绘制两种心形曲线
    heart_curve1()
    heart_curve2()
    
    # 设置图形属性
    plt.title('Heart Curves')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # 保存图形而不是显示
    plt.savefig('heart_curve.png')
    plt.close()
    print("图形已保存为 'heart_curve.png'")

def plot_animated_heart():
    """绘制动态心形曲线"""
    print("抱歉，在当前环境下无法显示动态图形。")
    return

if __name__ == "__main__":
    print("选择要显示的心形曲线类型：")
    print("1. 静态心形曲线")
    print("2. 动态心形曲线")
    
    choice = input("请输入选项（1或2）：")
    
    if choice == '1':
        plot_heart_curve()
    elif choice == '2':
        plot_animated_heart()
    else:
        print("无效的选项！")