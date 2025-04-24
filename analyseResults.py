import os
import re
import math
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
        plt.figure(figsize=(8,6))
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
            plt.figure(figsize=(8,6))
            for lab, grp in grp_model.groupby('label'):
                grp_sorted = grp.sort_values('k')
                xs, ys, se = grp_sorted['k'], grp_sorted['acc'], grp_sorted['se']
                plt.plot(xs, ys, '-o', label=lab)
                plt.fill_between(xs, ys-1.96*se, ys+1.96*se, alpha=0.2)
            plt.xlabel('Number of Distractors (k)')
            plt.ylabel('Accuracy')
            plt.title(f'Model: {model}')
            plt.ylim(0,1)
            plt.legend(title='Label', bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
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
        out_lines = []

        # 1) report total vs correct (valid) per label/model
        for (lab, mod), group in df.groupby(['label', 'model']):
            total = len(group)
            invalid = group['is_invalid'].sum()
            correct = total - invalid
            acc_pct = (correct / total * 100) if total else 0
            out_lines.append(
                f"Label={lab}  Model={mod}  "
                f"Total={total}  "
                f"Correct={correct} ({acc_pct:.2f}%)"
            )

        # 2) now compute mean±SE on the valid rows only
        valid = df[~df['is_invalid']]
        stats = (
            valid
            .groupby(['label', 'model', 'num_distractors'])['euclidean_error']
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_error'})
        )
        stats['se'] = stats['std'] / np.sqrt(stats['count'])

        return ({'error_stats': stats}, "\n".join(out_lines) + "\n")

    def plot(self, metrics):
        stats = metrics['error_stats']

        # Combined plot with 95% CI
        plt.figure(figsize=(10, 6))
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
            plt.figure(figsize=(10, 6))
            for lab, grp in grp_model.groupby('label'):
                grp = grp.sort_values('num_distractors')
                xs = grp['num_distractors']
                ys = grp['avg_error']
                se = grp['se']
                plt.plot(xs, ys, '-o', label=lab)
                plt.fill_between(xs, ys - 1.96 * se, ys + 1.96 * se, alpha=0.2)
            plt.xlabel('Number of Distractors')
            plt.ylabel('Average Euclidean Error')
            plt.title(f'Model: {model}')
            plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()


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
