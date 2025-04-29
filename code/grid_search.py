import subprocess

# 定义超参数的搜索空间
types = ['none']
probs = [0.]
view_types = ['flatten']

# 遍历所有超参数组合
for type_ in types:
    for prob in probs:
        for view_type in view_types:
                    # 构建命令
            command = [
                'python', 'main.py',
                '--type', type_,
                '--prob', str(prob),
                '--view_type', view_type
            ]
            print(f"Running command: {' '.join(command)}")
            try:
                # 执行命令
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running command: {e}")