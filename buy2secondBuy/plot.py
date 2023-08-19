import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='Times New Roman')

def read_log_file(log_file):
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []
    test_accs = []
    val_f1_scores = []
    val_tprs = []
    val_fprs = []
    val_aucs = []
    TPs = []
    FPs = []
    TNs = []
    FNs = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\]: In epoch (\d+), train loss: ([\d.]+), val loss: ([\d.]+), val acc: ([\d.]+) \(best [\d.]+\), test acc: ([\d.]+) \(best [\d.]+\), val F1: ([\d.]+), val TPR: ([\d.]+), val FPR: ([\d.]+), val AUC: ([\d.]+), TP: ([\d]+), FP: ([\d]+), TN: ([\d]+), FN: ([\d]+)', line)

            if match:
                epochs.append(int(match.group(2)))
                train_losses.append(float(match.group(3)))
                val_losses.append(float(match.group(4)))
                val_accs.append(float(match.group(5)))
                test_accs.append(float(match.group(6)))
                val_f1_scores.append(float(match.group(7)))
                val_tprs.append(float(match.group(8)))
                val_fprs.append(float(match.group(9)))
                val_aucs.append(float(match.group(10)))
                # TPs.append(int(match.group(11)))
                # FPs.append(int(match.group(12)))
                # TNs.append(int(match.group(13)))
                # FNs.append(int(match.group(14)))
                continue
    log_dict = {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_accs': test_accs,
        'val_f1_scores': val_f1_scores,
        'val_tprs': val_tprs,
        'val_fprs': val_fprs,
        'val_aucs': val_aucs,
        # 'TPs': TPs,
        # 'FPs': FPs,
        # 'TNs': TNs,
        # 'FNs': FNs
    }

    return log_dict

def read_log_and_plot(log_file):
    """
    绘制模型的训练曲线
    """
    # 绘制训练损失曲线和验证损失曲线
    model_name = re.findall(r'training_log_(.+).txt', log_file)[0]
    # epochs, train_losses, val_losses, val_accs, test_accs, val_f1_scores, val_tprs, val_fprs, val_aucs, TPs, FPs, TNs, FNs = read_log_file(log_file)
    log_dict = read_log_file(log_file)

    # 绘制训练损失曲线和验证损失曲线
    f, axes = plt.subplots(nrows = 5, figsize=(10,15))
    axes[0].plot(log_dict['epochs'], log_dict['train_losses'], alpha=0.5, label='training loss')
    axes[0].plot(log_dict['epochs'], log_dict['val_losses'], alpha=0.5, label='validation loss')
    axes[0].set_title(model_name)
    # axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # 绘制验证准确率曲线和测试准确率曲线
    axes[1].plot(log_dict['epochs'], log_dict['val_accs'], alpha=0.5, label='Validation Accuracy')
    # axes[1].plot(epochs, test_accs, alpha=0.5, label='Test Accuracy')
    # axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # 绘制验证F1 Score曲线
    axes[2].plot(log_dict['epochs'], log_dict['val_f1_scores'], alpha=0.5,label='Validation F1 Score')
    # axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()

    # 绘制验证TPR和FPR曲线
    f = plt.figure(figsize=(10, 2))
    axes[3].plot(log_dict['epochs'], log_dict['val_tprs'],alpha=0.5,  label='Validation TPR')
    axes[3].plot(log_dict['epochs'], log_dict['val_fprs'],alpha=0.5, label='Validation FPR')
    # axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Rate')
    axes[3].legend()

    # 绘制验证AUC曲线
    axes[4].plot(log_dict['epochs'], log_dict['val_aucs'],alpha=0.5, label='Validation AUC')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('AUC')
    axes

if __name__=='__main__':
    # 读取训练日志文件
    log_gcn = read_log_file('./logs/GCN/training_log_GCN_L3_LR0.01_WD0.0005_ESNone.txt')
    log_chebynet = read_log_file('./logs/ChebyNet/training_log_ChebyNet_L3_LR0.01_WD0.0005_ESNone.txt')
    log_gat = read_log_file('./logs/GAT/training_log_GAT_L3_LR0.01_WD0.0005_ESNone.txt')
    log_sage = read_log_file('./logs/GraphSAGE/training_log_GraphSAGE_L3_LR0.01_WD0.0005_ESNone.txt')
    log_tag = read_log_file('./logs/TAGGCN/training_log_TAGGCN_L3_LR0.01_WD0.0005_ESNone.txt')

    # 1. 绘制训练损失曲线
    f = plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(log_gcn['epochs'], log_gcn['train_losses'], label='GCN')
    plt.plot(log_chebynet['epochs'], log_chebynet['train_losses'], label='ChebyNet')
    plt.plot(log_sage['epochs'], log_sage['train_losses'], label='GraphSage')
    plt.plot(log_gat['epochs'], log_gat['train_losses'], label='GAT')
    plt.plot(log_tag['epochs'], log_tag['train_losses'], label='TAGConv')
    plt.legend()
    plt.xlim(0, 2000)
    plt.ylim(0.6, 0.85)
    plt.xlabel('Training step', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=16)
    plt.savefig('train_losses.pdf', bbox_inches='tight')

    # 2. 绘制验证损失曲线
    f = plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(log_gcn['epochs'], log_gcn['val_losses'], label='GCN')
    plt.plot(log_chebynet['epochs'], log_chebynet['val_losses'], label='ChebyNet')
    plt.plot(log_sage['epochs'], log_sage['val_losses'], label='GraphSage')
    plt.plot(log_gat['epochs'], log_gat['val_losses'], label='GAT')
    plt.plot(log_tag['epochs'], log_tag['val_losses'], label='TAGConv')
    plt.legend()
    plt.xlim(0, 2000)
    plt.ylim(0.6, 0.85)
    plt.xlabel('Training step', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=16)
    plt.savefig('val_losses.pdf', bbox_inches='tight')

    # 3. 绘制真阳性和假阳性群体的特征分布图
    df = pd.read_csv('data_pred.csv')
    f, axes = plt.subplots(figsize=(15, 6), ncols=5, nrows=2, dpi=300)

    for i, feature in enumerate(df.columns[:-1]):
        ax = axes[i // 5, i % 5]
        kde = sns.kdeplot(data=df, x=feature, hue='LABEL', ax=ax)
        legend = kde.legend_
        handles = legend.legendHandles
        labels = [t.get_text() for t in legend.get_texts()]
        ax.legend(handles=handles, labels=labels, ncol=1, fontsize='small')
        ax.set_xlabel(xlabel=feature, fontsize=18)
        ax.set_ylabel(ylabel='Density', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    axes[2, 3].set_frame_on(False)
    axes[2, 3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[2, 4].set_frame_on(False)
    axes[2, 4].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig('kde.pdf')
    plt.show()

