import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def load_data(file_list):
    all_data = []
    for file in file_list:
        if 'modality' in file:
            continue
        tensor_data = torch.load(file)
        try:
            all_data.append(tensor_data.detach().cpu().numpy())
        except AttributeError:
            all_data.append(tensor_data)
    return np.concatenate(all_data) if len(all_data) > 1 else all_data[0]


def plot_uncertainty_distributions(base_dir, datasets, aggregations, aggregation_labels, output_filename):
    num_datasets = len(datasets)
    num_aggregations = len(aggregations)
    fig, axes = plt.subplots(num_datasets, num_aggregations, figsize=(num_aggregations * 3, num_datasets * 2))

    for row, dataset in enumerate(datasets):
        for col, aggregation in enumerate(aggregations):
            lamb = '1.0' if aggregation in ['doc', 'weighted_doc'] else '0.1'
            activation = 'exp' if aggregation in ['doc', 'weighted_doc'] else 'softplus'
            file_normal = f'{base_dir}/{aggregation}_{dataset}_uncertainty_values_normal_{activation}_{lamb}.pth'
            file_conflict = f'{base_dir}/{aggregation}_{dataset}_uncertainty_values_conflict_{activation}_{lamb}.pth'
            try:
                normal_uncertainty = torch.load(file_normal).detach().cpu().numpy()
                conflict_uncertainty = torch.load(file_conflict).detach().cpu().numpy()
            except FileNotFoundError:
                print(f'File not found: {file_normal} or {file_conflict}')
                continue

            ax = axes[row, col]
            sns.kdeplot(conflict_uncertainty.reshape(-1), color='#ff7f0e', label='Conflict', ax=ax, fill=True)
            sns.kdeplot(normal_uncertainty.reshape(-1), color='#1f77b4', label='Normal', ax=ax, fill=True)
            ax.set_xlim([0, 1])
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(axis='both', which='minor', labelsize=17)
            ax.grid(False)
            uncertainties = np.concatenate([normal_uncertainty, conflict_uncertainty])
            labels = np.concatenate([np.zeros_like(normal_uncertainty), np.ones_like(conflict_uncertainty)])
            fpr, tpr, _ = roc_curve(labels, uncertainties)
            roc_auc = auc(fpr, tpr)
            ax.text(0.75, 0.9, f'AUC: {roc_auc:.2f}', fontsize=15, ha='center', va='center', transform=ax.transAxes)
            if row == 0:
                ax.set_title(aggregation_labels[aggregation], fontsize=17, weight='bold' if col == 2 else 'normal')
            if row == num_datasets - 1:
                ax.set_xlabel('Uncertainty', fontsize=17)
            if col == 0:
                ax.set_ylabel('Density', fontsize=17)
            else:
                ax.set_ylabel('')

    for row, dataset in enumerate(datasets):
        y_position = 0.19 + (len(datasets) - row - 1) * 0.175
        if row == 0:
            y_position += 0.02
        fig.text(0.5, y_position, f'{dataset}', ha='center', va='bottom', fontsize=18, weight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=15, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(wspace=None, hspace=0.5)
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_lambda_uncertainty(base_dir, dataset, agg, lambdas, output_filename):
    fig, axes = plt.subplots(1, len(lambdas), figsize=(10, 1))

    for i, l in enumerate(lambdas):
        activation = 'exp' if agg in ['doc', 'weighted_doc'] else 'softplus'
        file_normal = f'{base_dir}/{agg}_{dataset}_uncertainty_values_normal_{activation}_{l:.1f}.pth'
        file_conflict = f'{base_dir}/{agg}_{dataset}_uncertainty_values_conflict_{activation}_{l:.1f}.pth'
        try:
            normal_uncertainty = torch.load(file_normal).detach().cpu().numpy()
            conflict_uncertainty = torch.load(file_conflict).detach().cpu().numpy()
        except FileNotFoundError:
            print(f'File not found: {file_normal} or {file_conflict}')
            continue

        ax = axes[i]
        sns.kdeplot(conflict_uncertainty.reshape(-1), color='#ff7f0e', label='Conflict', ax=ax, fill=True)
        sns.kdeplot(normal_uncertainty.reshape(-1), color='#1f77b4', label='Normal', ax=ax, fill=True)
        ax.set_xlim([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis='both', which='minor', labelsize=17)
        ax.grid(False)
        ax.set_title(rf'$\lambda=${l}', fontsize=17)
        ax.set_xlabel('Uncertainty', fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=13)
        if i == 4:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_ylabel('Density' if i == 0 else '', fontsize=17)
    fig.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)


def plot_average_modality_uncertainties(base_dir, datasets, aggregation, output_filename):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(3, 2, figsize=(6, 4), sharey='all', sharex=False)

    for ax, dataset in zip(axes.flatten(), datasets):
        lamb = '1.0'
        activation = 'exp'
        modality_uncertainty = []
        for i in range(10):
            file = f'{base_dir}/{aggregation}_{dataset}_modality_uncertainty_values_normal_{activation}_{lamb}_run_{i}.pth'
            try:
                mu = torch.load(file).detach().cpu().numpy()
                modality_uncertainty.append(mu.mean(axis=0).squeeze())
            except FileNotFoundError:
                continue
        modality_uncertainty = np.stack(modality_uncertainty)
        data = pd.DataFrame(modality_uncertainty, columns=[f'M{i + 1}' for i in range(modality_uncertainty.shape[-1])])
        sns.barplot(data, ax=ax, palette='viridis')
        ax.set_yticks(np.arange(0, 1.1, 0.5))
        ax.set_title(f'{dataset}', fontsize=10)
        ax.set_ylim([0, 1])
        ax.grid(False)

    plt.subplots_adjust(wspace=None, hspace=0.55)
    for row in range(3):
        axes[row, 0].set_ylabel('Avg. Uncertainty')
    axes[2, 0].set_xlabel('Modalities')
    axes[1, -1].set_xlabel('Modalities')
    axes[2, 1].axis('off')
    fig.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)


def generate_latex_table(exp_results_dir):
    df = pd.concat([pd.read_csv(f'{exp_results_dir}/{i}') for i in os.listdir(exp_results_dir) if i.endswith('.csv')])
    df = df[df['aggregation'] != 'weighted_doc']
    grouped_df = df.groupby(['dataset', 'aggregation']).agg({'auc': 'mean', 'auc_std': 'mean'}).reset_index()

    # Create "mean ± std" formatted string for each row
    grouped_df['mean_std'] = grouped_df['auc'].map('{:.2f}'.format) + ' ± ' + grouped_df['auc_std'].map('{:.2f}'.format)

    # Pivot the DataFrame to get datasets as columns and methods as rows
    pivot_formatted_df = grouped_df.pivot(index='aggregation', columns='dataset', values='mean_std')

    # Generate LaTeX table using booktabs style
    latex_table_code = pivot_formatted_df.to_latex(
        index=True,
        caption="Mean AUC and Standard Deviation for Different Aggregations by Dataset",
        label="tab:auc_results",
        escape=False,
        bold_rows=True,
        column_format="lcccc",
        header=True,
        position='htbp',
        multicolumn=True,
        multicolumn_format='c',
        # booktabs=True
    )

    print(latex_table_code)


# Example usage
if __name__ == "__main__":
    datasets = ['CalTech', 'HandWritten', 'CUB', 'PIE', 'Scene']
    aggregations = ['tmc', 'conf_agg', 'doc']
    aggregation_labels = {
        'sum': 'CBF',
        'tmc': 'BCF',
        'conf_agg': 'BAF',
        'doc': 'DBF (our)',
        'weighted_doc': 'WDBF (our)'
    }

    plot_uncertainty_distributions('exp_results', datasets, aggregations, aggregation_labels,
                                   'conflict_normal_uncertainty_distributions.pdf')
    plot_lambda_uncertainty('sms_exp_results', 'PIE', 'doc', [0.2, 0.5, 1, 2, 3][::-1], 'lambda_unc.pdf')
    plot_average_modality_uncertainties('exp_results', datasets, 'doc', 'average_modality_uncertainties.pdf')
    generate_latex_table('exp_results')