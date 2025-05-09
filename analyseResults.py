import os
import re
import math
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import rcParams

sns.set(style="whitegrid")


class Analysis:
    """
    Base class for different analysis modes.
    """
    def __init__(self, directories, labels, groups=None, output_dir=None, save_figs=False, confusion=False):
        # Basic args
        self.directories = directories
        self.labels = labels
        self.save_figs = save_figs
        self.confusion = confusion

        # Output directory
        if output_dir:
            self.output_dir = output_dir
        elif save_figs and len(directories) == 1:
            self.output_dir = os.path.join("results", directories[0])
        else:
            self.output_dir = None

        # map dirs to labels
        if len(directories) != len(labels):
            raise ValueError("The number of directories and labels must match.")
        self.dir_label = dict(zip(directories, labels))

        # parse group specs
        self.group_map = {}
        if groups:
            for spec in groups:
                grp, members = spec.split(':', 1)
                for d in members.split(','):
                    self.group_map[d] = grp

        # text output file name
        self.text_filename = f"{self.mode}_analysis_results.txt"

    @property
    def mode(self):
        """Return mode string; override in subclasses."""
        raise NotImplementedError

    @property
    def file_suffix(self):
        """Suffix of files to load; override in subclasses."""
        raise NotImplementedError

    def load_csvs(self):
        """
        Walk through directories, load only files ending with the given suffix.
        Attach label and model to each row.
        """
        dfs = []
        for dire in self.directories:
            label = self.dir_label[dire]
            dpath = os.path.join("results", dire)
            files = [f for f in os.listdir(dpath) if f.endswith(self.file_suffix)]
            if not files:
                raise FileNotFoundError(f"No files ending with '{self.file_suffix}' in {dpath}")

            for fname in files:
                path = os.path.join(dpath, fname)
                df = pd.read_csv(path)
                model = fname.replace(self.file_suffix, '')

                # tag
                df['label'] = label
                df['model'] = model
                df['source_dir'] = dire
                df['source_file'] = fname
                dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        return combined

    def preprocess(self, df):
        """Default no-op preprocess."""
        return df

    def compute_metrics(self, df):
        """Compute metrics; return (metrics_dict, text_output)."""
        raise NotImplementedError

    def plot(self, metrics):
        """Generate plots from metrics dict."""
        raise NotImplementedError

    def save_text(self, text):
        """Save analysis text to file or stdout."""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            outpath = os.path.join(self.output_dir, self.text_filename)
        else:
            outpath = os.path.join(os.getcwd(), self.text_filename)

        with open(outpath, 'w') as f:
            f.write(text)
        print(text)
        print(f"Written analysis text to {outpath}")

    def run(self):
        df = self.load_csvs()
        df = self.preprocess(df)
        metrics, text = self.compute_metrics(df)
        self.save_text(text)
        self.plot(metrics)


class CellAnalysis(Analysis):
    @property
    def mode(self):
        return 'cell'
    
    @property
    def file_suffix(self):
        return '_results_Cells.csv'

    def parse_cell(self, cell_str):
        m = re.match(r"\((\d+),\s*(\d+)\)", str(cell_str))
        return (int(m.group(1)), int(m.group(2))) if m else None

    def preprocess(self, df):
        # rename for consistency
        df = df.rename(columns={'selected_cell': 'predicted_cell'})

        # parse both actual and predicted into tuples
        df['actual_cell_parsed']    = df['actual_cell'].apply(self.parse_cell)
        df['predicted_cell_parsed'] = df['predicted_cell'].apply(self.parse_cell)

        # flag invalid predictions
        df['is_invalid'] = df['predicted_cell_parsed'].isna()

        # wherever invalid, mark correct=False
        df.loc[df['is_invalid'], 'correct'] = False
        
        # keep any pre-existing correctness flag for valid rows
        # (assumes your CSV had a 'correct' column for valid ones)

        # apply grouping labels
        df['label'] = df['source_dir'].map(self.group_map).fillna(df['label'])

        # map numerical cells to human‐readable labels
        unique = sorted(set(df['actual_cell_parsed']))
        label_map = {c: f'Cell ({c[0]}, {c[1]})' for c in unique}
        df['actual_label']    = df['actual_cell_parsed'].map(label_map)
        df['predicted_label'] = df['predicted_cell_parsed'].map(lambda c: label_map.get(c, 'Invalid'))

        return df

    def compute_metrics(self, df):
        lines = []

        # 1) for each label/model, report total, invalid and accuracy
        for (lab, mod), g in df.groupby(['label','model']):
            total         = len(g)
            invalid_count = g['is_invalid'].sum()
            acc_count     = g['correct'].sum()
            invalid_pct   = invalid_count/total*100 if total else 0
            acc_pct       = acc_count/total*100     if total else 0

            lines.append(
                f"Label={lab}  Model={mod}  "
                f"Total={total}  "
                f"Invalid={invalid_count} ({invalid_pct:.2f}%)  "
                f"Acc={acc_pct:.2f}%"
            )

            # 2) detailed classification report, but only for true classes
            classes = sorted(g['actual_label'].unique())
            lines.append(
                classification_report(
                    g['actual_label'],
                    g['predicted_label'],
                    labels=classes,
                    zero_division=0
                )
            )

            if self.confusion:
                cm = confusion_matrix(
                    g['actual_label'],
                    g['predicted_label'],
                    labels=classes+['Invalid']
                )
                cm_df = pd.DataFrame(cm, index=classes+['Invalid'], columns=classes+['Invalid'])
                lines.append(f"Confusion Matrix:\n{cm_df}\n")

        # 3) build the per-cell success rates and accuracies vs k as before
        succ = []
        for (lab,mod), g in df.groupby(['label','model']):
            for cell, grp in g.groupby('actual_label'):
                succ.append({
                    'label': lab,
                    'model': mod,
                    'cell': cell,
                    'success': (grp['predicted_label']==cell).mean()
                })
        succ_df = pd.DataFrame(succ)

        avs = []
        for (lab,mod), g in df.groupby(['label','model']):
            for k, grp in g.groupby('num_distractors'):
                acc = grp['correct'].mean()
                se  = math.sqrt(acc*(1-acc)/len(grp)) if len(grp) > 1 else 0
                avs.append({'label':lab,'model':mod,'k':k,'acc':acc,'se':se})
        avs_df = pd.DataFrame(avs)

        return (
            {'success_df': succ_df, 'avs_df': avs_df},
            "\n\n".join(lines) + "\n"
        )

    def plot(self, metrics):
        succ_df = metrics['success_df']
        avs_df = metrics['avs_df']

        # Combined accuracy vs distractors per label-model
        plt.figure(figsize=figSize)
        for (lab,mod), grp in avs_df.groupby(['label','model']):
            xs, ys, se = grp['k'], grp['acc'], grp['se']
            plt.plot(xs,ys,'-o',label=f"{lab}-{mod}")
            plt.fill_between(xs, ys-1.96*se, ys+1.96*se, alpha=0.2)
        plt.xlabel('Number of Distractors (k)')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.legend(title='Label - Model', bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Individual plots per model, comparing labels
        for model, grp_model in avs_df.groupby('model'):
            plt.figure(figsize=figSize)
            print(model)
            for lab, grp in grp_model.groupby('label'):
                grp_sorted = grp.sort_values('k')
                xs, ys, se = grp_sorted['k'], grp_sorted['acc'], grp_sorted['se']
                plt.plot(xs, ys, '-o', label=lab)
                plt.fill_between(xs, ys-1.96*se, ys+1.96*se, alpha=0.2)
            plt.xlabel('Number of Distractors (k)')
            plt.ylabel('Accuracy')
            #plt.title(f'Model: {model}')
            plt.ylim(0,1)
            plt.legend(title='Label', bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            plt.show()


        selected_models = ['gpt-4o', 'claude-sonnet', 'llama90B'] 
        modelTitleMap = {
            'gpt-4o': 'GPT-4o',
            'claude-sonnet': 'Claude Sonnet',
            'llama90B': 'Llama 90B',
            'claude-haiku': 'Claude Haiku',
        }


        # bump up font for paper
        plt.rcParams.update({'font.size': 18})

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

        for ax, model in zip(axes.flatten(), selected_models):
            grp_model = avs_df[avs_df['model'] == model]
            if grp_model.empty:
                ax.set_title(f"{model} (no data)")
                continue

            for label, grp in grp_model.groupby('label'):
                grp = grp.sort_values('k')
                prettyMod = modelTitleMap.get(model, model)
                ax.plot(grp['k'], grp['acc'], '-o', label=label)
                ax.fill_between(grp['k'],
                                grp['acc'] - 1.96*grp['se'],
                                grp['acc'] + 1.96*grp['se'],
                                alpha=0.2)
            prettyMod = modelTitleMap.get(model, model)
            ax.set_title(prettyMod)
            ax.set_xlabel('Number of Distractors (k)')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0,1)


        for ax in axes[0]:
                ax.set_xlabel('')
        for ax in axes[:, 1]:
            ax.set_ylabel('')

        for ax in axes[1, :]:
            ax.set_xlabel('Number of Distractors (k)', fontsize=18)

        # grab handles/labels from any one subplot
        handles, labels = axes[0, 0].get_legend_handles_labels()

        # place single legend below the grid, in figure‐coords
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_transform=fig.transFigure,
            bbox_to_anchor=(0.5, 0.02),  # x=0.5 centers, y≈0.02 just above bottom
            ncol=len(labels),
            fontsize=20,
            frameon=False
        )

        # now tighten the layout, leaving room at bottom for legend
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

class CoordsAnalysis(Analysis):
    @property
    def mode(self):
        return 'coords'

    @property
    def file_suffix(self):
        return '_results_Coords.csv'

    def preprocess(self, df):
        # flag rows with no error as invalid
        df['is_invalid'] = df['euclidean_error'].isna()

        # convert distractor counts to int
        df['num_distractors'] = pd.to_numeric(df['num_distractors'], errors='coerce').astype(int)

        # apply grouping labels
        df['label'] = df['source_dir'].map(self.group_map).fillna(df['label'])
        return df

    
    def compute_metrics(self, df):
        # clamp invalid errors to the image diagonal
        maxError = np.hypot(400, 400)  # ≈565.7 px
        df = df.copy()
        df.loc[df['is_invalid'], 'euclidean_error'] = maxError

        # 1) report total vs correct (valid) per label/model
        outLines = []
        for (lab, mod), groupDf in df.groupby(['label', 'model']):
            total = len(groupDf)
            invalid = groupDf['is_invalid'].sum()
            correct = total - invalid
            accPct = (correct / total * 100) if total else 0
            outLines.append(
                f"Label={lab}  Model={mod}  "
                f"Total={total}  "
                f"Valid={correct} ({accPct:.2f}%)"
            )

        # 2) compute mean ± SE on all rows (invalid ones carry maxError)
        errorStats = (
            df
            .groupby(['label', 'model', 'num_distractors'])['euclidean_error']
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_error'})
        )
        errorStats['se'] = errorStats['std'] / np.sqrt(errorStats['count'])

        return ({'error_stats': errorStats}, "\n".join(outLines) + "\n")

    def plot(self, metrics):

        df_all = self.load_csvs()
        df_all = self.preprocess(df_all)
        df_all['is_valid'] = ~df_all['is_invalid']

        # get valid‐rate by label+model
        rates = (
            df_all
            .groupby(['label', 'model'])['is_valid']
            .mean()
            .reset_index()
        )
        rates['accuracy_pct'] = rates['is_valid'] * 100

        # grab the same color cycle you use in line plots
        color_cycle = rcParams['axes.prop_cycle'].by_key()['color']

        # map each label to one color in that cycle
        labels = sorted(rates['label'].unique())
        color_map = {lab: color_cycle[i % len(color_cycle)] 
                     for i, lab in enumerate(labels)}

        # one bar‐chart per model
        for model, grp in rates.groupby('model'):
            print(model)
            plt.figure(figsize=(8, 5))
            # assign each bar its label’s color
            bar_colors = [color_map[lab] for lab in grp['label']]
            plt.bar(grp['label'], grp['accuracy_pct'], color=bar_colors)
            plt.xlabel('Label')
            plt.ylabel('Valid Rate (%)')
            #plt.title(f'Valid/Correct Rate for Model: {model}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if self.save_figs and self.output_dir:
                fname = f'coords_valid_rate_{model}.png'
                plt.savefig(os.path.join(self.output_dir, fname))
            plt.show()



        stats = metrics['error_stats']

        # Combined plot with 95% CI
        plt.figure(figsize=figSize)
        for (lab, mod), grp in stats.groupby(['label', 'model']):
            grp = grp.sort_values('num_distractors')
            xs = grp['num_distractors']
            ys = grp['avg_error']
            se = grp['se']
            plt.plot(xs, ys, '-o', label=f"{lab}-{mod}")
            plt.fill_between(xs, ys - 1.96 * se, ys + 1.96 * se, alpha=0.2)
        plt.xlabel('Number of Distractors')
        plt.ylabel('Average Euclidean Error')
        plt.legend(title='Label - Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Individual plots per model
        for model, grp_model in stats.groupby('model'):
            print(model)
            plt.figure(figsize=figSize)
            for lab, grp in grp_model.groupby('label'):
                grp = grp.sort_values('num_distractors')
                xs = grp['num_distractors']
                ys = grp['avg_error']
                se = grp['se']
                plt.plot(xs, ys, '-o', label=lab)
                plt.fill_between(xs, ys - 1.96 * se, ys + 1.96 * se, alpha=0.2)
            plt.xlabel('Number of Distractors')
            plt.ylabel('Average Euclidean Error')
            plt.ylim(bottom=0)
            #plt.title(f'Model: {model}')
            plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()


        selected_models = ['gpt-4o', 'claude-sonnet','llama90B'] 
        modelTitleMap = {
            'gpt-4o': 'GPT-4o',
            'claude-sonnet': 'Claude Sonnet',
            'llama90B': 'Llama 90B',
            'claude-haiku': 'Claude Haiku',
        }

        plt.rcParams.update({'font.size': 18})


        # --- 2×2 SUBPLOT PANEL FOR FOUR MODELS ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

        for ax, model in zip(axes.flatten(), selected_models):
            pretty = modelTitleMap.get(model, model)
            grp = stats[stats['model'] == model].sort_values('num_distractors')
            if grp.empty:
                ax.set_title(f"{pretty}\n(no data)")
                continue

            for rawLab, sub in grp.groupby('label'):
                ax.plot(sub['num_distractors'], sub['avg_error'], '-o', label=rawLab)
                ax.fill_between(
                    sub['num_distractors'],
                    sub['avg_error'] - 1.96*sub['se'],
                    sub['avg_error'] + 1.96*sub['se'],
                    alpha=0.2
                )
            ax.set_title(pretty)
            ax.set_xlabel('')   # clear by default—you’ll restore only bottom
            ax.set_ylabel('')   # clear by default—only left will show

        # clear top x-labels, right y-labels
        for ax in axes.flatten():
            ax.set_ylabel('')

            # add one figure-level y-label (centered alongside the left column)
            # if you have Matplotlib ≥3.4:
            fig.supylabel(
                'Average Euclidean Error',
                x=0.032,       # tweak horizontal position;  
                fontsize=24,
                va='center'
            )

        # restore bottom & left labels
        for ax in axes[1]:
            ax.set_xlabel('Number of Distractors')
       
        # single legend below
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_transform=fig.transFigure,
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(labels),
            fontsize=20,
            frameon=False
        )

        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()
        # ---------------------------------------------



class PresenceAnalysis(Analysis):
    @property
    def mode(self): return 'presence'
    @property
    def file_suffix(self): return '_results_Presence.csv'

    def preprocess(self, df):
        df['sel'] = pd.to_numeric(df['selected_presence'],errors='coerce')
        df['act'] = pd.to_numeric(df['actual_presence'],errors='coerce')
        df['correct'] = df['sel']==df['act']
        # apply grouping labels
        df['label'] = df['source_dir'].map(self.group_map).fillna(df['label'])
        return df

    def compute_metrics(self, df):
        out=''
        avs=[]
        for (lab,mod), g in df.groupby(['label','model']):
            out += f"Label={lab} Model={mod} Total={len(g)} Acc={g['correct'].mean()*100:.2f}%\n"
            out += classification_report(g['act'],g['sel'],zero_division=0) + '\n'
            for k,grp in g.groupby('num_distractors'):
                acc=grp['correct'].mean(); se=math.sqrt(acc*(1-acc)/len(grp)) if len(grp)>1 else 0
                avs.append({'label':lab,'model':mod,'k':k,'acc':acc,'se':se})
        return ({'avs_df':pd.DataFrame(avs)}, out)

    def plot(self, metrics):
        df=metrics['avs_df']
        # Combined
        plt.figure(figsize=(8,6))
        for (lab,mod), grp in df.groupby(['label','model']):
            grp_sorted = grp.sort_values('k')
            plt.plot(grp_sorted['k'],grp_sorted['acc'],'-o',label=f"{lab}-{mod}")
            plt.fill_between(grp_sorted['k'],grp_sorted['acc']-1.96*grp_sorted['se'],grp_sorted['acc']+1.96*grp_sorted['se'],alpha=0.2)
        plt.xlabel('Number of Distractors (k)')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.legend(title='Label - Model', bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Individual plots per model
        for model, grp_model in df.groupby('model'):
            plt.figure(figsize=(8,6))
            for lab, grp in grp_model.groupby('label'):
                grp_sorted = grp.sort_values('k')
                plt.plot(grp_sorted['k'],grp_sorted['acc'],'-o',label=lab)
                plt.fill_between(grp_sorted['k'],grp_sorted['acc']-1.96*grp_sorted['se'],grp_sorted['acc']+1.96*grp_sorted['se'],alpha=0.2)
            plt.xlabel('Number of Distractors (k)')
            plt.ylabel('Accuracy')
            plt.title(f'Model: {model}')
            plt.ylim(0,1)
            plt.legend(title='Label', bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m','--mode',choices=['cell','coords','presence'],required=True)
    p.add_argument('-d','--directories',nargs='+',required=True)
    p.add_argument('-l','--labels',nargs='+',required=True)
    p.add_argument('-g','--groups',nargs='+')
    p.add_argument('-c','--confusion',action='store_true')
    args = p.parse_args()


    sns.set_context("talk", font_scale=1.3)
    figSize=(14,8)



    # determine save_figs
    save_figs = len(args.directories)==1
    common_kwargs = dict(
        directories=args.directories,
        labels=args.labels,
        groups=args.groups,
        save_figs=save_figs,
        confusion=args.confusion
    )
    if args.mode=='cell':
        runner = CellAnalysis(**common_kwargs)
    elif args.mode=='coords':
        runner = CoordsAnalysis(**common_kwargs)
    else:
        runner = PresenceAnalysis(**common_kwargs)

    runner.run()
