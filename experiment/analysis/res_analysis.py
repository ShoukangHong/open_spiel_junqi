import re
import matplotlib.pyplot as plt
import pandas as pd


def analyze_report(file):
    data = {
        'step': [],
        'total': [],
        'policy': [],
        'value': [],
        'l2': []
    }

    with open(file, 'r') as f:
        lines = f.readlines()

    step = 0
    for line in lines:
        if "Losses(" in line:
            total_match = re.search(r'total:\s*([0-9.]+)', line)
            policy_match = re.search(r'policy:\s*([0-9.]+)', line)
            value_match = re.search(r'value:\s*([0-9.]+)', line)
            l2_match = re.search(r'l2:\s*([0-9.]+)', line)

            if total_match and policy_match and value_match and l2_match:
                data['step'].append(step)
                data['total'].append(float(total_match.group(1)))
                data['policy'].append(float(policy_match.group(1)))
                data['value'].append(float(value_match.group(1)))
                data['l2'].append(float(l2_match.group(1)))
                step += 1

    # 单图显示所有曲线
    plt.figure(figsize=(10, 6))
    plt.plot(data['step'], data['total'], label='Total Loss', linewidth=2)
    plt.plot(data['step'], data['policy'], label='Policy Loss', linewidth=2)
    plt.plot(data['step'], data['value'], label='Value Loss', linewidth=2)
    plt.plot(data['step'], data['l2'], label='L2 Loss', linewidth=2)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Losses Over Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return data

if __name__ == '__main__':
    data = analyze_report(r"C:\Users\shouk\Github\open_spiel_junqi\temp\records\res.txt")
    df = pd.DataFrame(data)
    print(df)