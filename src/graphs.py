import matplotlib.pyplot as plt
import seaborn as sns

def categorical_vs_target_rep(df, columns, target, n_col):

    rows = int(round(n_col/2))

    fig, ax = plt.subplots(nrows=rows, ncols=2, figsize=(12, 10))
    ax = ax.flatten()

    for i, col in enumerate(columns):
        grpd_df = df.groupby([col, f'{target}']).size().reset_index(name='count')
        bar_plot = grpd_df.pivot(col, f'{target}', 'count').plot(kind='bar', ax=ax[i])
            
        ax[i].set_title(f'Count of {target} instances for each gategory of {col}')
        ax[i].set_xlabel(col)
        ax[i].set_ylabel('Number of instances')
        ax[i].grid()
            
        for p in ax[i].patches:
            ax[i].annotate(str(p.get_height()), (p.get_x() + 0.01, p.get_height() + 0.6)) 
            
    if n_col % 2 != 0:
        fig.delaxes(ax[-1])
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def model_comparison_heatmap(cm1, cm2, score1, score2):
    
    fig = plt.figure(figsize = (13, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy of model improved by gridsearchCV: {0}'.format(score1)
    plt.title(all_sample_title, size = 12)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy of model improved manually: {0}'.format(score2)
    plt.title(all_sample_title, size = 12)

    plt.tight_layout()
    plt.show()