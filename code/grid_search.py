import subprocess

# 定义超参数的搜索空间
types = ['none']
probs = [0.1]
view_types = ['mean']
enable_genre = False
time_stamp = False
model_name = 'BSARec'

# 遍历所有超参数组合
for type_ in types:
    for prob in probs:
        for view_type in view_types:
                    # 构建命令
            command = [
                'python', 'main.py',
                '--type', type_,
                '--prob', str(prob),
                '--view_type', view_type,
                '--model', model_name,
            ]
            if enable_genre:
                command.append('--genre') 
            if time_stamp:
                command.append('--timestamp')
            print(f"Running command: {' '.join(command)}")
            try:
                # 执行命令
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running command: {e}")