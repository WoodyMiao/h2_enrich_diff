#!/usr/bin/env python

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import t, sem

if len(sys.argv) == 1:
    sys.exit(f'''
**************************************
* Tests h2med enrichment differences *
* (C) 2021 Lin Miao @ Miaoxin Li Lab *
* MIT License                        *
**************************************

To use this script, mesc/sumstats.py should be modified to output regression coefficients. The objects of hsqhat.coef 
and hsqhat.part_delete_values are ndarrays of whole data coefficients and jackknife coefficients respectively. Use 
np.savetxt('MESC_OUT_PREFIX.GENE_SETx.coef_jackknife', hsqhat.part_delete_values, comments='') and 
np.savetxt('MESC_OUT_PREFIX.GENE_SETx.coef_wholedata', hsqhat.coef, comments='') to output them with suffixes of 
.GENE_SET1.coef_wholedata, .GENE_SET1.coef_jackknife, .GENE_SET2.coef_wholedata and .GENE_SET2.coef_jackknife.

Show help: {sys.argv[0]} -h 
''')

parser = argparse.ArgumentParser()
parser.add_argument('--gannot-chr', default=None, type=str, help='Prefix of MESC annotation files')
parser.add_argument('--mesc-out', default=None, type=str, help='Prefix of MESC output files.')
parser.add_argument('--gene-set1', default=None, type=str, help='Name of the 1st gene set for enrichment comparison.')
parser.add_argument('--gene-set2', default=None, type=str, help='Name of the 2st gene set for enrichment comparison.')
parser.add_argument('--num-bins', default=5, type=int,
                    help='Number of overall expression cis-heritability bins. Default 5.')
parser.add_argument('--num-gene-bins', default=3, type=int,
                    help='Number of expression cis-heritability bins per gene set. Default 3.')
parser.add_argument('--out', default=None, type=str, help='The output file. Default to MESC_OUT.enrich_diff_p.')
parser.add_argument('--header', default=None, type=str, help='Header in the output file. Default to MESC_OUT.')


def se_jackknife(theta, theta_jk):
    # theta (p,) or scalar
    # theta_jk (200, p) or (200,)
    n = theta_jk.shape[0]
    pseudovalues = n * theta - (n - 1) * theta_jk  # (200, p) or (200,)
    return sem(pseudovalues)  # (p,) or ()


def per_gene_h2med_diff_p(gene_set1_h2med, gene_set1_h2med_jk, gene_set1_ngene,
                          gene_set2_h2med, gene_set2_h2med_jk, gene_set2_ngene, dof):
    per_gene_h2med_diff = gene_set1_h2med / gene_set1_ngene - gene_set2_h2med / gene_set2_ngene
    per_gene_h2med_diff_jk = gene_set1_h2med_jk / gene_set1_ngene - gene_set2_h2med_jk / gene_set2_ngene
    per_gene_h2med_diff_se = se_jackknife(per_gene_h2med_diff, per_gene_h2med_diff_jk)
    per_gene_h2med_diff_z = per_gene_h2med_diff / per_gene_h2med_diff_se
    per_gene_h2med_diff_p_ = t.sf(abs(per_gene_h2med_diff_z), dof) * 2
    return per_gene_h2med_diff, per_gene_h2med_diff_se, per_gene_h2med_diff_p_


class GeneSet:
    def __init__(self, gene_set_name):
        coef_wholedata = np.loadtxt(f'{args.mesc_out}.{gene_set_name}.coef_wholedata')
        coef_jackknife = np.loadtxt(f'{args.mesc_out}.{gene_set_name}.coef_jackknife')
        self.coef = coef_wholedata[-num_exp_score_coef:]
        self.coef_jk = coef_jackknife[:, -num_exp_score_coef:]

        self.gene_cols = [f'{gene_set_name}_Cis_herit_bin_{i + 1}' for i in range(args.num_gene_bins)]
        self.mesc_cols = overall_cols + self.gene_cols
        self.ngene = gannot[self.gene_cols].values.sum()
        self.h2med, self.h2med_jk = self.cal_h2med(self.gene_cols)

    def h2med_enrich_p(self):
        overall_h2med, overall_h2med_jk = self.cal_h2med(overall_cols)
        complement_h2med = overall_h2med - self.h2med
        complement_h2med_jk = overall_h2med_jk - self.h2med_jk
        complement_ngene = overall_ngene - self.ngene
        return per_gene_h2med_diff_p(self.h2med, self.h2med_jk, self.ngene, complement_h2med,
                                     complement_h2med_jk, complement_ngene, self.coef_jk.shape[0]-1)

    def cal_h2med(self, cols_to_cal):
        h2med = 0
        h2med_jk = np.zeros(self.coef_jk.shape[0], )  # (200,)
        for col in cols_to_cal:
            g = gannot.loc[gannot[col] == 1, self.mesc_cols].sum().values  # (8,)
            h2med += (g * ave_h2cis[self.mesc_cols].values * self.coef).sum()
            h2med_jk += (g * ave_h2cis[self.mesc_cols].values * self.coef_jk).sum(axis=1).reshape(-1)
        return h2med, h2med_jk


def info_pre():
    a = datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
    return f'INFO  {a} - '


if __name__ == '__main__':
    args = parser.parse_args()
    print('''
**************************************
* Tests h2med enrichment differences *
* (C) 2021 Lin Miao @ Miaoxin Li Lab *
* MIT License                        *
**************************************
''')
    print(f'{info_pre()}Read gene annotations.')
    try:
        gannot = pd.concat(
            map(lambda c: pd.read_csv(f'{args.gannot_chr}.{c}.gannot.gz', sep='\t', index_col=0), range(1, 23)))
    except FileNotFoundError:
        gannot = pd.concat(
            map(lambda c: pd.read_csv(f'{args.gannot_chr}.{c}.gannot', sep='\t', index_col=0), range(1, 23)))

    G = np.array([np.loadtxt(f'{args.gannot_chr}.{c}.G') for c in range(1, 23)])
    ave_h2cis = np.array([np.loadtxt(f'{args.gannot_chr}.{c}.ave_h2cis') for c in range(1, 23)])
    ave_h2cis = pd.Series(np.average(ave_h2cis, axis=0, weights=G), index=gannot.columns)

    overall_cols = [f'Cis_herit_bin_{i + 1}' for i in range(args.num_bins)]
    overall_ngene = gannot[overall_cols].values.sum()
    num_exp_score_coef = args.num_bins + args.num_gene_bins

    print(f'{info_pre()}Calculate p-values of h2med enrichments.')
    gene_set1 = GeneSet(args.gene_set1)
    gene_set2 = GeneSet(args.gene_set2)
    set1_, set1_se, set1_p = gene_set1.h2med_enrich_p()
    set2_, set2_se, set2_p = gene_set2.h2med_enrich_p()
    diff_, diff_se, diff_p = per_gene_h2med_diff_p(gene_set1.h2med, gene_set1.h2med_jk, gene_set1.ngene, gene_set2.h2med,
                                                   gene_set2.h2med_jk, gene_set2.ngene, gene_set2.h2med_jk.shape[0]-1)

    results = pd.Series(
        [set1_, set1_se, set1_p, set2_, set2_se, set2_p, diff_, diff_se, diff_p],
        name=args.header if args.header else os.path.basename(args.mesc_out),
        index=pd.MultiIndex.from_product([[args.gene_set1, args.gene_set2, 'Between'],
                                          ['perGENEdiff', 'perGENEdiff_se', 'perGENEdiff_p']]))
    out_file = args.out if args.out else f'{args.mesc_out}.enrich_diff_p'
    results.to_csv(out_file, sep='\t')
    print(f'{info_pre()}Results wrote to {out_file}\n')
