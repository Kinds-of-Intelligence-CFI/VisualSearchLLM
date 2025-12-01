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
from matplotlib.ticker import MultipleLocator, MaxNLocator

import itertools


sns.set(style="whitegrid")


class Analysis:
    """
    Base class for different analysis modes.
    """
    def __init__(self, directories, labels, groups=None, output_dir=None, save_figs=False, confusion=False, experiment=None):
        # Basic args
        self.directories = directories
        self.labels = labels
        self.save_figs = save_figs
        self.confusion = confusion
        self.experiment=experiment

        # Output directory
        if output_dir:
            self.output_dir = output_dir
        elif save_figs:
            if len(directories) == 1:
                self.output_dir = os.path.join("results", directories[0])
            else:
                self.output_dir = os.path.join("results", "plots")
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

    def __init__(self, directories, labels, groups=None, output_dir=None,
                     save_figs=False, confusion=False, experiment=None, humanData=False):
        super().__init__(directories, labels, groups, output_dir, save_figs, confusion, experiment)
        
        self.humanData = humanData
        if self.humanData:
            
            experimentCSVMap = {"2Among5": "e1_numbers_processed", "LightPriors": "e2_light_priors_processed", "CircleSizes":"e3_circle_sizes_processed"}

            experimentFile = experimentCSVMap[self.experiment]
            df = pd.read_csv(os.path.join(f"humanResults/{experimentFile}.csv"))
            #df = clean_columns(df)
            if self.experiment == "2Among5":
                condition = "colour_type"
                df['colour_type'] = df['colour_type'].replace({
                    'Inefficient disjunctive': 'Shape Conjunctive',
                    'Efficient disjunctive': 'Disjunctive',
                    'Conjunctive': 'Shape-Colour Conjunctive'
                })

                bin_edges = [1, 5, 9, 17, 33, 65, 100]
                bin_labels = ['1–4','5–8','9–16','17–32','33–64','65–99']

            elif self.experiment == "LightPriors":

                condition = "light_direction"
                
                

                #df=df[(df["light_direction"]=="bottom") | (df["light_direction"]=="left")]

                df["light_direction"] = df["light_direction"].replace({"left": "Left", "bottom": "Bottom", "right": "Right", "top": "Top"})
                #print(df[condition])
                bin_edges = [1, 6, 11, 15, 18] 
                bin_labels = ['1–5','6–10','11–14', '15–17']
            elif self.experiment == "CircleSizes": 
                condition = "target_size"
                bin_edges = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 50]
                bin_labels = ['1–4','5–8','9–12','13–16','17–20','21–24','25-28','29-32','33-36','37-40','41-44','45-49']
               
            df['distractor_bin'] = pd.cut(df['num_distractors'],
                                          bins=bin_edges,
                                          labels=bin_labels,
                                          right=False)

            ## remove low scoring participants
            participant_accuracy = df.groupby('PID')['accuracy'].mean()
            accuracy_threshold = 0.25
            valid_participants = participant_accuracy[participant_accuracy >= accuracy_threshold].index
            total_participants = len(participant_accuracy)
            filtered_participants = total_participants - len(valid_participants)
            print(f"Total participants: {total_participants}")
            print(f"Participants below threshold ({accuracy_threshold*100}%): {filtered_participants}")
            print(f"Participants remaining: {len(valid_participants)}")
            self.human_df = df[df['PID'].isin(valid_participants)]
        else:
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
        df['is_invalid'] = ~df['predicted_cell_parsed'].isin([(1,1),(1,2),(2,1),(2,2)])

        # wherever invalid, mark correct=False
        df.loc[df['is_invalid'], 'correct'] = False
        
        # keep any pre-existing correctness flag for valid rows
        # (assumes your CSV had a 'correct' column for valid ones)

        # apply grouping labels
        df['label'] = df['source_dir'].map(self.group_map).fillna(df['label'])

        # map numerical cells to human‐readable labels
        unique = sorted([x for x in set(df['actual_cell_parsed']) if x is not None])
        label_map = {c: f'Cell ({c[0]}, {c[1]})' for c in unique}
        df['actual_label']    = df['actual_cell_parsed'].map(label_map)
        df['predicted_label'] = df['predicted_cell_parsed'].map(lambda c: label_map.get(c, 'Invalid'))

        return df

    def compute_metrics(self, df):
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


        # ---------------------------------------------------------------------
        # Build a text summary *grouped by model*
        # ---------------------------------------------------------------------
        lines = []
        for mod in sorted(df['model'].unique()):
            lines.append(f"Model: {mod}")
            df_mod = df[df['model']==mod]

            for lab in sorted(df_mod['label'].unique()):
                g = df_mod[df_mod['label']==lab]
                total         = len(g)
                invalid_count = g['is_invalid'].sum()
                acc_count     = g['correct'].sum()
                invalid_pct   = (invalid_count/total*100) if total else 0
                acc_pct       = (acc_count/total*100)     if total else 0

                # summary line
                lines.append(
                    f"  Label={lab}  Total={total}  "
                    f"Invalid={invalid_count} ({invalid_pct:.2f}%)  "
                    f"Acc={acc_pct:.2f}%"
                )

                # classification report
                classes = sorted(g['actual_label'].dropna().unique())
                if not classes:
                    lines.append("    No valid actual labels found.")
                else:
                    y_true = g['actual_label'].fillna('Unknown').astype(str)
                    y_pred = g['predicted_label'].fillna('Unknown').astype(str)
                    report = classification_report(
                        y_true,
                        y_pred,
                        labels=[str(c) for c in classes],
                        zero_division=0
                    )
                    for rpt_line in report.splitlines():
                        lines.append(f"    {rpt_line}")
                pred_counts = g['predicted_label'].value_counts(normalize=True)
                pred_props = {label: pred_counts.get(label, 0) for label in classes}
                pred_prop_line = "  Predicted Proportions: " + "  ".join(
                    f"{label}={prop*100:.1f}%" for label, prop in pred_props.items()
                )
                lines.append(pred_prop_line)
            lines.append("")  # blank line between models
        text_output = "\n".join(lines) + "\n"
        metrics = {
            'success_df': succ_df,
            'avs_df':     avs_df,
        }
        return metrics, text_output



    def plot(self, metrics):

        if self.humanData:
            featureMap = {"2Among5": 'colour_type', "LightPriors":'light_direction', 'CircleSizes': 'target_size'}
            raw = (
                self.human_df
                  .rename(columns={featureMap[self.experiment]:'label','distractor_bin':'bin'})
                  .groupby(['label','bin'], observed=True)['accuracy']
                  .agg(['mean','std','count'])
                  .reset_index()
                  .rename(columns={'mean':'acc'})
            )
         
            raw['se'] = raw['std'] / np.sqrt(raw['count'])

            # normalize & split bin-strings robustly
            bin_str = raw['bin'].astype(str) \
                         .str.replace('–','-',regex=False) \
                         .str.strip()
            lo_hi = bin_str.str.split('-', expand=True)
            lo_hi.columns = ['lo','hi']
            lo_hi['lo'] = pd.to_numeric(lo_hi['lo'], errors='coerce')
            lo_hi['hi'] = pd.to_numeric(lo_hi['hi'], errors='coerce')
            valid = lo_hi['lo'].notna() & lo_hi['hi'].notna()
            raw = raw[valid].reset_index(drop=True)
            lo_hi = lo_hi[valid].reset_index(drop=True)

            # compute mid-points
            raw['x'] = (lo_hi['lo'] + lo_hi['hi']) / 2

            humanStats = raw
        else:
            humanStats=None

        succ_df = metrics['success_df']
        avs_df = metrics['avs_df']


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
            ax = plt.gca()
            kmin, kmax = int(grp_model['k'].min()), int(grp_model['k'].max())
            # If the span is small, show every integer; otherwise, just ensure integers
            if kmax - kmin <= 20:
                ax.xaxis.set_major_locator(MultipleLocator(1))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(
                loc='lower center',
                bbox_to_anchor=(0.5, -0.25),
                ncol=len(grp_model.groupby('label')),
                fontsize=20,
                frameon=False
            )
            plt.tight_layout()
            if self.save_figs and self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                fname = f'cell_accuracy_{model}.png'
                plt.savefig(os.path.join(self.output_dir, fname))
            plt.show()


        if self.experiment=="2Among5":
            edges = np.array([1, 5, 9, 17, 33, 65, 100])   
            binLabels = ['1–4','5–8','9–16','17–32','33–64','65–99'] 
        elif self.experiment=="LightPriors":
            edges = np.array([1, 5, 9, 13, 17, 21, 25, 33, 50])
            binLabels = ['1–4','5–8','9–12','13–16','17–20','21–24','25-32','33-50']
        elif self.experiment== "CircleSizes":
            edges = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 50])
            binLabels=['1–4','5–8','9–12','13–16','17–20','21–24','25-28','29-32','33-36','37-40','41-44','45-49']

        mids       = 0.5 * (edges[:-1] + edges[1:])
        label2mid  = dict(zip(binLabels, mids))

        for model, grp_model in avs_df.groupby('model'):
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

            plt.figure(figsize=figSize)
            print(model)
            for lab, g in agg.groupby('label'):
                xs, ys, se = g['k_mid'], g['acc_mean'], g['acc_se']
                plt.plot(xs, ys, '-o', label=lab)
                plt.fill_between(xs, ys - 1.96*se, ys + 1.96*se, alpha=0.2)

            plt.xlabel('Number of Distractors (k, binned)')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.xticks(mids, binLabels,  rotation=45, ha='right')             # centres on the mids, text from your list
            plt.legend(
                loc='lower center',
                bbox_to_anchor=(0.5, -0.4),
                ncol=len(grp_model.groupby('label')),
                fontsize=20,
                frameon=False
            )
            plt.tight_layout()
            if self.save_figs and self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                fname = f'cell_accuracy_binned_{model}.png'
                plt.savefig(os.path.join(self.output_dir, fname))
            plt.show()

        ### 2x2 plot for paper

        selected_models = ['gpt-4o', 'claude-sonnet', 'llama90B', 'Qwen7B', 'Qwen32B'] 
        modelTitleMap = {
            'gpt-4o': 'GPT-4o',
            'claude-sonnet': 'Claude Sonnet',
            'llama90B': 'Llama 90B',
            'Qwen7B': 'Qwen 7B',
            'Qwen32B': 'Qwen 32B',
        }


        # bump up font for paper
        plt.rcParams.update({'font.size': 18})

        fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharex=False, sharey=True)



        avs_df['label'] = avs_df['label'].str.title()
        if humanStats is not None:
            humanStats['label'] = humanStats['label'].str.title()



        all_labels = sorted(
            set(avs_df['label'].unique())
            | set(humanStats['label'].unique() if humanStats is not None else [])
        )
        cycle = rcParams['axes.prop_cycle'].by_key()['color']
        color_map = {lbl: cycle[i % len(cycle)] for i, lbl in enumerate(all_labels)}




        for ax, model in zip(axes.flatten()[:len(selected_models)], selected_models):
            grp_model = avs_df[avs_df['model'] == model]
            if grp_model.empty:
                ax.set_title(f"{model} (no data)")
                continue

            for label, grp in grp_model.groupby('label'):
                grp = grp.sort_values('k')
                prettyMod = modelTitleMap.get(model, model)
           
                ax.plot(
                       grp['k'], grp['acc'],
                       '-o',
                       color=color_map[label],
                       label=label
                )
                ax.fill_between(
                    grp['k'],
                    grp['acc'] - 1.96*grp['se'],
                    grp['acc'] + 1.96*grp['se'],
                    color=color_map[label],
                    alpha=0.2
                )
            prettyMod = modelTitleMap.get(model, model)
            ax.set_title(prettyMod)
            ax.set_xlabel('Number of Distractors (k)')
            ax.set_ylabel('Accuracy')
            #ax.set_xlim(0,99)
            ax.set_ylim(0,1)
            #tick_positions = [0, 20, 40, 60, 80, 99]
            
            if  self.experiment == "CircleSizes":
                tick_positions = [0,10,20,30,40,49]
            elif self.experiment == "2Among5":
                tick_positions = [0,20,40,60,80,99]
            elif self.experiment == "LightPriors":
                tick_positions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            else:
                tick_positions = [0, 20, 40, 60, 80, 99]
            for ax in axes.flatten()[:len(selected_models)]:
                #ax.set_xlim(0, 99)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(t) for t in tick_positions])

        if self.humanData:
            axH = axes[1,1]
            for hlabel, hgrp in humanStats.groupby('label'):
                hgrp = hgrp.sort_values('x')
    
                axH.plot(
                    hgrp['x'], hgrp['acc'],
                    '-o',
                    color=color_map[hlabel],
                    linewidth=2,
                    label=hlabel
                )
                axH.fill_between(
                    hgrp['x'],
                    hgrp['acc'] - 1.96*hgrp['se'],
                    hgrp['acc'] + 1.96*hgrp['se'],
                    color=color_map[hlabel],
                    alpha=0.2
                )


            xticks = hgrp['x'].unique()  # or sorted(humanStats['x'].unique())
            xlabels = raw['bin'].unique()  # same order as xticks after sorting
        # to be safe:
            order = np.argsort(xticks)
            xticks = xticks[order]
            xlabels = np.array(raw['bin'].unique())[order]

            axH.set_xticks(xticks)
            axH.set_xticklabels(xlabels, rotation=45, ha='right')

            axH.set_title('Human')
            axH.set_xlabel('Number of Distractors (k)')
            axH.set_ylabel('Accuracy')
            axH.set_ylim(0,1)


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
        if self.save_figs and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            fname = f'cell_accuracy_combined.png'
            plt.savefig(os.path.join(self.output_dir, fname))
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
        df['is_outside'] = (df['selected_x'] > 400) | (df['selected_y'] > 400)

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


        outside_rate = (
            df_all
            .groupby(['label', 'model'])['is_outside'].mean().reset_index()
            )
        outside_rate["outside_pct"]=outside_rate['is_outside']*100
        print(outside_rate)

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

        # one bar‐chart per model
        for model, grp in outside_rate.groupby('model'):
            print(model)
            plt.figure(figsize=(8, 5))
            # assign each bar its label’s color
            bar_colors = [color_map[lab] for lab in grp['label']]
            plt.bar(grp['label'], grp['outside_pct'], color=bar_colors)
            plt.xlabel('Label')
            plt.ylabel('Outside Rate (%)')
            #plt.title(f'Valid/Correct Rate for Model: {model}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if self.save_figs and self.output_dir:
                fname = f'coords_outside_rate_{model}.png'
                plt.savefig(os.path.join(self.output_dir, fname))
            plt.show()
        stats = metrics['error_stats']

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

            ax = plt.gca()
            dmin, dmax = int(grp_model['num_distractors'].min()), int(grp_model['num_distractors'].max())
            if dmax - dmin <= 20:
                ax.xaxis.set_major_locator(MultipleLocator(1))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.legend(
                loc='lower center',
                bbox_to_anchor=(0.5, -0.25),
                ncol=len(grp_model.groupby('label')),
                fontsize=20,
                frameon=False
            )
            plt.tight_layout()
            plt.show()

        selected_models = ['gpt-4o', 'claude-sonnet','llama90B'] 
        modelTitleMap = {
            'gpt-4o': 'GPT-4o',
            'claude-sonnet': 'Claude Sonnet',
            'llama90B': 'Llama 90B'
        }

        plt.rcParams.update({'font.size': 18})


        fig, axes = plt.subplots(
           1, len(selected_models),
           figsize=(18, 5),      # wider than tall
           sharey=True
        )

        for ax, model in zip(axes, selected_models):
           pretty = modelTitleMap.get(model, model)
           grp   = stats[stats['model'] == model].sort_values('num_distractors')

           if grp.empty:
               ax.set_title(f"{pretty}\n(no data)")
               continue

           for lab, sub in grp.groupby('label'):
               ax.plot(
                   sub['num_distractors'],
                   sub['avg_error'],
                   '-o',
                   label=lab
               )
               ax.fill_between(
                   sub['num_distractors'],
                   sub['avg_error'] - 1.96*sub['se'],
                   sub['avg_error'] + 1.96*sub['se'],
                   alpha=0.2
               )

           ax.set_title(pretty)
           ax.set_xlabel('Number of Distractors')
           
           # only left‐most keeps the y‐label
           if ax is axes[0]:
               ax.set_ylabel('Average Euclidean Error')
           else:
               ax.set_ylabel('')
           ax.set_ylim(bottom=0)


           ylimMap = {"2Among5": 350, "LightPriors":300 , "CircleSizes":450}
           ax.set_ylim(0, ylimMap[self.experiment])
           if self.experiment == "2Among5":
                tick_positions = [0, 20, 40, 60, 80, 99]

           elif self.experiment in ["CircleSizes"]:
                tick_positions = [0,10,20,30,40,49]
           elif self.experiment == "LightPriors":
                tick_positions = [2,4,6,8,10,12,14,16]
           



           for ax in axes.flatten()[:len(selected_models)]:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(t) for t in tick_positions], ha='right')

        # single legend below all three
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
           handles, labels,
           loc='lower center',
           bbox_transform=fig.transFigure,
           bbox_to_anchor=(0.5, -0.03),
           ncol=len(labels),
           fontsize=16,
           frameon=False
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m','--mode',choices=['cell','coords'],required=True)
    p.add_argument('-d','--directories',nargs='+',required=True)
    p.add_argument('-l','--labels',nargs='+',required=True)
    p.add_argument('-g','--groups',nargs='+')
    p.add_argument("-e", "--experiment", choices=['2Among5', 'LightPriors', 'CircleSizes'], required=True)
    p.add_argument("--humanData", action='store_true')
    p.add_argument('-c','--confusion',action='store_true')
    p.add_argument('-s', '--save', action='store_true', help='Save figures to disk')
    p.add_argument('-o', '--output_dir', help='Directory to save results')
    args = p.parse_args()


    sns.set_context("talk", font_scale=1.3)
    figSize=(14,8)



    # determine save_figs
    save_figs = args.save or len(args.directories)==1
    common_kwargs = dict(
        directories=args.directories,
        labels=args.labels,
        groups=args.groups,
        save_figs=save_figs,
        confusion=args.confusion,
        experiment = args.experiment,
        output_dir = args.output_dir,
    )

    if args.mode=='cell':
        common_kwargs["humanData"]=args.humanData
        runner = CellAnalysis(**common_kwargs)
    elif args.mode=='coords':
        runner = CoordsAnalysis(**common_kwargs)

    runner.run()
