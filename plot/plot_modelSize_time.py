# -*- coding: utf-8 -*-
# @Time    : 2024/10/5 22:37
# @Author  : Li Bo
# @FileName: ssss.py
# @Software: PyCharm

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据
data = {
    'Model Size (MB)': [
        0.32, 0.16, 0.64, 0.66, 0.45, 0.71, 2.27, 7.87,
        4.67, 0.18, 0.55, 1.80, 4.30, 286.45, 9.58, 0.13
    ],
    'Fusion Time (S)': [
        0.29, 1.52, 1.98, 1.0, 1.26, 0.55, 1.39, 2.71,
        74.48, 0.37, 5.05, 0.42, 0.47, 14.35, 1.24, 0.19
    ],
    'Method': [
        'IFCNN', 'PMGI', 'CU-Net', 'U2Fusion',
        'MFF-GAN', 'SDNet', 'SwinFusion', 'DeFusion',
        'ZMFF', 'MGDN', 'MUFusion', 'PSLPT',
        'DB-MFIF', 'DeepM2CDL', 'TC-MoA', 'NSNPFuse (Ours)'
    ]
}

# 数据转为 DataFrame
df = pd.DataFrame(data)

with plt.style.context(['ieee']):

    plt.figure(figsize=(6, 4))  # 增大图的尺寸
    scatter_plot = sns.scatterplot(
        data=df,
        x='Model Size (MB)',
        y='Fusion Time (S)',
        s=100,
        hue='Method',  # 按方法不同设置颜色
        palette="bright",  # 选择调色板
        legend=False , # 不显示图例
        # marker = '2',  # 五角星形状
    )

    # 设置双对数坐标
    plt.xscale('log')  # 对横轴取对数
    plt.yscale('log')  # 对纵轴取对数

    # 设置标题和标签
    plt.xlabel("Model Size (MB)", fontsize=14)
    plt.ylabel("Fusion Time (S)", fontsize=14)

    # 获取 'NSNPFuse (Ours)' 的索引并绘制为红色五角星
    idx_ours = df[df['Method'] == 'NSNPFuse (Ours)'].index[0]
    plt.scatter(
        df['Model Size (MB)'][idx_ours],
        df['Fusion Time (S)'][idx_ours],
        color='red',  # 红色
        s=350,  # 增大点的大小
        marker='*',  # 五角星形状
        label='NSNPFuse (Ours)'  # 标签
    )

    # 在每个点显示方法名称，避免超出画布
    for i in range(len(df)):
        x, y = df['Model Size (MB)'][i], df['Fusion Time (S)'][i]
        method = df['Method'][i]
        if method == 'DeepM2CDL':
            method = r'DeepM$^{2}$CDL'  # 使用 LaTeX 格式显示上标
        if x > 100:  # 根据点的横坐标调整标注偏移
            plt.annotate(
                method, (x, y),
                xytext=(-15, 5),  # 左上偏移
                textcoords="offset points",
                ha='right',
                fontsize=10
            )
        else:
            plt.annotate(
                method, (x, y),
                xytext=(5, 5),  # 右上偏移
                textcoords="offset points",
                ha='left',
                fontsize=10
            )

    # 自动调整边距避免文字溢出
    plt.tight_layout()

    # 保存图片
    plt.savefig("scatter_double_log_with_red_star.jpg", dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.show()


