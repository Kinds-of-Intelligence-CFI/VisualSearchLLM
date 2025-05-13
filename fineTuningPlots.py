"""
A quickly hacked script based on resultsAnalysis.py for the paper fine-tuning plot (side-by-side 2among5s and binned TamongLs)

Generated with e.g.
python fineTuningPlots.py -dl "ft-dataset-0-test", "ft-dataset-10-test", "ft-dataset-100-test", "ft-dataset-1000-990-test" -dr "ft-dataset-0-ts-ls-test" "ft-dataset-10-ts-ls-test" "ft-dataset-100-ts-ls-test" "ft-dataset-1000-ts-ls-test" -l n=0 n=10 n=100 n=1000 -m cell -e 2Among5
"""

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
    def __init__(self, directoriesLeft, directoriesRight, labels, groups=None, output_dir=None, save_figs=False, confusion=False, experiment=None):
        # Basic args
        self.directoriesLeft = directoriesLeft
        self.directoriesRight = directoriesRight
        self.labels = labels
        self.save_figs = save_figs
        self.confusion = confusion
        self.experiment=experiment

        # Output directory
        if output_dir:
            self.output_dir = output_dir
        elif save_figs and len(directoriesLeft) == 1:
            self.output_dir = os.path.join("results", directoriesLeft[0])
        else:
            self.output_dir = None

        # map dirs to labels
        if len(directoriesLeft) != len(labels):
            raise ValueError("The number of directories and labels must match.")
        self.dir_label_left = dict(zip(directoriesLeft, labels))
        self.dir_label_right = dict(zip(directoriesRight, labels))

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
        for dire in self.directoriesLeft:
            label = self.dir_label_left[dire]
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

        combined_left = pd.concat(dfs, ignore_index=True)

        dfs = []
        for dire in self.directoriesRight:
            label = self.dir_label_right[dire]
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

        combined_right = pd.concat(dfs, ignore_index=True)
        return combined_left, combined_right

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
        df_left, df_right = self.load_csvs()
        df_left = self.preprocess(df_left)
        df_right = self.preprocess(df_right)
        metrics_left, _ = self.compute_metrics(df_left)
        metrics_right, _ = self.compute_metrics(df_right)
        self.plot(metrics_left, metrics_right)


class CellAnalysis(Analysis):

    def __init__(self, directoriesLeft, directoriesRight, labels, groups=None, output_dir=None,
                     save_figs=False, confusion=False, experiment=None, humanData=False):
        super().__init__(directoriesLeft, directoriesRight, labels, groups, output_dir, save_figs, confusion, experiment)
        
        self.human_experiment=None
        self.human_df=None


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

        # 3) build the per-cell success rates and accuracies vs k 
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

    def plot(self, metricsLeft, metricsRight):

        ### 1x2 plot for paper

        # bump up font for paper
        plt.rcParams.update({'font.size': 18})

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False, sharey=True)
        ax_left = axes[0]
        ax_right = axes[1]

        humanStats=None

        avs_df_left = metricsLeft['avs_df']
        avs_df_right = metricsRight['avs_df']

        avs_df_left['label'] = avs_df_left['label'].str.title()
        avs_df_right['label'] = avs_df_right['label'].str.title()



        all_labels = sorted(
            set(avs_df_left['label'].unique())
        )
        cycle = rcParams['axes.prop_cycle'].by_key()['color']
        color_map = {lbl: cycle[i % len(cycle)] for i, lbl in enumerate(all_labels)}

        # Subplot 1
        # Individual plots per model, comparing labels
        for model, grp_model in avs_df_left.groupby('model'):
            # plt.figure(figsize=figSize)
            print(model)
            for lab, grp in grp_model.groupby('label'):
                grp_sorted = grp.sort_values('k')
                xs, ys, se = grp_sorted['k'], grp_sorted['acc'], grp_sorted['se']
                ax_left.plot(xs, ys, '-o', label=lab)
                ax_left.fill_between(xs, ys-1.96*se, ys+1.96*se, alpha=0.2)
            ax_left.set_xlabel('Number of Distractors (k)')
            ax_left.set_ylabel('Accuracy')
            #plt.title(f'Model: {model}')
            ax_left.set_ylim(0,1)
            ax_left.set_title("2-among-5s")

        # Subplot 2
        if self.experiment=="2Among5":
            edges = np.array([1, 5, 9, 17, 33, 65, 100])   
            binLabels = ['1–4','5–8','9–16','17–32','33–64','65–99'] 
        elif self.experiment=="Light Priors":
            edges = np.array([1, 5, 9, 13, 17, 21, 25, 33, 50])
            binLabels = ['1–4','5–8','9–12','13–16','17–20','21–24','25-32','33-49']
        elif self.experiment== "CircleSizes":
            edges = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 50])
            ['1–4','5–8','9–12','13–16','17–20','21–24','25-28','29-32','33-36','37-40','41-44','45-49']

        mids       = 0.5 * (edges[:-1] + edges[1:])
        label2mid  = dict(zip(binLabels, mids))

        for model, grp_model in avs_df_right.groupby('model'):
            gm = grp_model.copy()
            
            # carve into categories that carry your custom labels
            gm['k_bin'] = pd.cut(gm['k'],
                                 bins=edges,
                                 labels=binLabels,
                                 right=False)

            # numeric x for plotting
            gm['k_mid'] = gm['k_bin'].map(label2mid)

            agg = (gm
                   .groupby(['label', 'k_bin'], observed=True)
                   .agg(acc_mean=('acc', 'mean'),
                        acc_se   =('acc', 'sem'),
                        k_mid    =('k_mid', 'first'))
                   .reset_index()
                   .sort_values('k_mid'))

            print(model)
            for lab, g in agg.groupby('label'):
                xs, ys, se = g['k_mid'], g['acc_mean'], g['acc_se']
                ax_right.plot(xs, ys, '-o', label=lab)
                ax_right.fill_between(xs, ys - 1.96*se, ys + 1.96*se, alpha=0.2)

            ax_right.set_xlabel('Number of Distractors (k, binned)')
            # ax_right.set_ylabel('Accuracy')
            ax_right.set_ylim(0, 1)
            ax_right.set_xticks(mids, binLabels,  rotation=45, ha='right')             # centres on the mids, text from your list
            ax_right.set_title("T-among-Ls, Binned")

        # grab handles/labels from any one subplot
        handles, labels = axes[0].get_legend_handles_labels()

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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m','--mode',choices=['cell'],required=True)
    p.add_argument('-dl','--directoriesLeft',nargs='+',required=True)
    p.add_argument('-dr','--directoriesRight',nargs='+',required=True)
    p.add_argument('-l','--labels',nargs='+',required=True)
    p.add_argument('-g','--groups',nargs='+')
    p.add_argument("-e", "--experiment", choices=['2Among5', 'LightPriors', 'CircleSizes'], required=True)
    p.add_argument("--humanData", action='store_true')
    p.add_argument('-c','--confusion',action='store_true')
    args = p.parse_args()


    sns.set_context("talk", font_scale=1.3)
    figSize=(14,8)



    # determine save_figs
    save_figs = len(args.directoriesLeft)==1
    common_kwargs = dict(
        directoriesLeft=args.directoriesLeft,
        directoriesRight=args.directoriesRight,
        labels=args.labels,
        groups=args.groups,
        save_figs=save_figs,
        confusion=args.confusion,
        experiment = args.experiment,

    )

    if args.mode=='cell':
        common_kwargs["humanData"]=args.humanData
        runner = CellAnalysis(**common_kwargs)
    else:
        raise ValueError("This script only works for cell")

    runner.run()
