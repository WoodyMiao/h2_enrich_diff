#!/usr/bin/env python

import os
import sys
import argparse
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import t, sem

if len(sys.argv) == 1:
    sys.exit(f'''
**************************************
* Testing h2g enrichment differences *
* (C) 2021 Lin Miao @ Miaoxin Li Lab *
* MIT License                        *
**************************************

When running LDSC, the flags of --overlap-annot, --print-coefficients, and --print-delete-vals are needed.

Show help: {sys.argv[0]} -h
''')

parser = argparse.ArgumentParser()
parser.add_argument('--annot-chr', default=None, type=str, help='Prefixes of annotation files, separated by comma.')
parser.add_argument('--frqfile-chr', default=None, type=str, help='Prefix of MAF files.')
parser.add_argument('--ldsc-out', default=None, type=str, help='Prefix of LDSC output files.')
parser.add_argument('--enrich-set1', default=None, type=str,
                    help='Names of the 1st sets of SNPs for enrichment comparison, separated by comma.')
parser.add_argument('--enrich-set2', default=None, type=str,
                    help='Names of the 2st sets of SNPs for enrichment comparison, separated by comma.')
parser.add_argument('--out', default=None, type=str, help='The output file. Default to LDSC_OUT.tsv')
parser.add_argument('--header', default=None, type=str, help='Header in the output file. Default to LDSC_OUT.')


def read_stdev():
    def func(y):
        stdev_file = f'{y}annot.stdev'
        try:
            std_ = pd.read_csv(stdev_file, sep='\t')
        except FileNotFoundError:
            print(f'{info_pre()}Calculate {stdev_file}')
            a = list()
            for c in range(1, 23):
                b = pd.read_csv(f'{y}{c}.annot.gz', sep='\t').iloc[:, 4:]
                a.append(b)
            std_ = pd.concat(a).std().to_frame().T
            std_.to_csv(stdev_file, sep='\t', index=False)
        return std_

    std_list = list()
    for x in annot_chr:
        std = func(x)
        std_list.append(std)
    std_all_annot = pd.concat(std_list, axis=1).loc[0]
    return std_all_annot


def read_annotate_chr(c):
    annot_list = list()
    for x in annot_chr:
        annot = pd.read_csv(f'{x}{c}.annot.gz', sep='\t').iloc[:, 4:]
        annot_list.append(annot)
    annot = pd.concat(annot_list, axis=1)
    maf = pd.read_csv(f'{args.frqfile_chr}{c}.frq', sep=r'\s+').MAF
    annot = annot[(maf >= 0.05)]
    n_snps_chr = annot.shape[0]
    all_sum_chr = annot.sum()
    part_sum_chr = pd.DataFrame({a: annot[annot[a] == 1].sum() for a in enrich_both})
    return n_snps_chr, all_sum_chr, part_sum_chr


def read_annotate():
    n_snp = 0
    all_sum = pd.Series(0, dtype=int, index=stdev.index)
    part_sum = pd.DataFrame(0, dtype=float, index=stdev.index, columns=enrich_both)
    for a, b, c in Pool(22).map(read_annotate_chr, range(1, 23)):
        n_snp += a
        all_sum += b
        part_sum += c
    return n_snp, all_sum, part_sum


def se_jackknife(theta, theta_jk):
    # theta (p,) or ()
    # theta_jk (200, p) or (200,)
    n = theta_jk.shape[0]
    pseudovalues = n * theta - (n - 1) * theta_jk  # (200, p) or (200,)
    return sem(pseudovalues)  # (p,) or ()


class OneTrait:
    def __init__(self, ldsc_out):
        ldsc_result = pd.read_csv(f'{ldsc_out}.results', sep='\t', index_col=0)
        self.ldsc_annots = ldsc_result.index.str.replace(r'L2_\d', '', regex=True).to_list()
        self.tau = ldsc_result.Coefficient.values
        self.H = np.sum(M_5_50[self.ldsc_annots].values * self.tau)
        self.H_jk = np.loadtxt(f'{ldsc_out}.delete')
        self.tau_jk = np.loadtxt(f'{ldsc_out}.part_delete')
        self.jk_nb = self.H_jk.shape[0]

    def cal_tau_star(self):
        # stdev (p,)
        # H ()
        # tau (p,)
        # H_jk (200, 1)
        # tau_jk (200, p)

        tau_star = M * stdev[self.ldsc_annots].values * self.tau / self.H  # (p,)
        tau_star_jk = M * stdev[self.ldsc_annots].values * self.tau_jk / self.H_jk.reshape(-1, 1)  # (200, p)
        tau_star_se = se_jackknife(tau_star, tau_star_jk)  # (p,)
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_star_z = tau_star / tau_star_se  # (p,)
            tau_star_p = t.sf(abs(tau_star_z), self.jk_nb - 1) * 2

        idx = pd.MultiIndex.from_product(
            [['Tau_star', 'Tau_star_se', 'Tau_star_z', 'Tau_star_p'], self.ldsc_annots]).swaplevel()
        return pd.Series(np.concatenate([tau_star, tau_star_se, tau_star_z, tau_star_p]), index=idx)[self.ldsc_annots]

    def cal_enr_diff(self):
        def cal_enr_p(nsnp_set1, nsnp_set2, h_set1, h_set2, h_set1_jk, h_set2_jk):
            # test heritability enrichment difference
            d = h_set1 / nsnp_set1 - h_set2 / nsnp_set2  # per-SNP heritability difference
            d_jk = h_set1_jk / nsnp_set1 - h_set2_jk / nsnp_set2  # (200,)
            d_se = se_jackknife(d, d_jk)
            d_z = d / d_se
            d_p = t.sf(abs(d_z), d_jk.shape[0] - 1) * 2
            return d, d_se, d_z, d_p

        def cal_enr(self_, a):
            # heritability of a
            h = np.sum(M_Enrich.loc[self_.ldsc_annots, a].values * self_.tau)
            h_jk = np.sum(M_Enrich.loc[self_.ldsc_annots, a].values * self_.tau_jk, axis=1)  # (200,)
            h_se = se_jackknife(h, h_jk)
            h_z = h / h_se
            h_p = t.sf(abs(h_z), self_.jk_nb - 1) * 2
            # heritability enrichment of a
            e = h / self_.H / M_5_50[a] * M
            e_jk = h_jk / self_.H_jk / M_5_50[a] * M  # (200,)
            e_se = se_jackknife(e, e_jk)
            d, d_se, d_z, d_p = cal_enr_p(M_5_50[a], M - M_5_50[a], h, self_.H - h, h_jk, self_.H_jk - h_jk)
            return h, h_jk, h_se, h_z, h_p, e, e_jk, e_se, d, d_se, d_z, d_p

        result_dict = dict()
        for a1, a2 in zip(enrich_set1, enrich_set2):
            h1, h1_jk, h1_se, h1_z, h1_p, e1, e1_jk, e1_se, d1, d1_se, d1_z, d1_p = cal_enr(self, a1)
            h2, h2_jk, h2_se, h2_z, h2_p, e2, e2_jk, e2_se, d2, d2_se, d2_z, d2_p = cal_enr(self, a2)
            e_ratio = e1 / e2
            e_ratio_jk = e1_jk / e2_jk
            e_ratio_se = se_jackknife(e_ratio, e_ratio_jk)
            diff, diff_se, diff_z, diff_p = cal_enr_p(M_5_50[a1], M_5_50[a2], h1, h2, h1_jk, h2_jk)
            update_dict = {
                (a1, 'h2'): h1,
                (a1, 'h2_se'): h1_se,
                (a1, 'h2_z'): h1_z,
                (a1, 'h2_p'): h1_p,
                (a1, 'nSNP'): M_5_50[a1],
                (a1, 'pSNP'): M_5_50[a1] / M,
                (a1, 'enrich'): e1,
                (a1, 'enrich_se'): e1_se,
                (a1, 'perSNPdiff'): d1,
                (a1, 'perSNPdiff_se'): d1_se,
                (a1, 'perSNPdiff_z'): d1_z,
                (a1, 'perSNPdiff_p'): d1_p,
                (a2, 'h2'): h2,
                (a2, 'h2_se'): h2_se,
                (a2, 'h2_z'): h2_z,
                (a2, 'h2_p'): h2_p,
                (a2, 'nSNP'): M_5_50[a2],
                (a2, 'pSNP'): M_5_50[a2] / M,
                (a2, 'enrich'): e2,
                (a2, 'enrich_se'): e2_se,
                (a2, 'perSNPdiff'): d2,
                (a2, 'perSNPdiff_se'): d2_se,
                (a2, 'perSNPdiff_z'): d2_z,
                (a2, 'perSNPdiff_p'): d2_p,
                (f'{a1}-{a2}', 'enrich'): e_ratio,
                (f'{a1}-{a2}', 'enrich_se'): e_ratio_se,
                (f'{a1}-{a2}', 'perSNPdiff'): diff,
                (f'{a1}-{a2}', 'perSNPdiff_se'): diff_se,
                (f'{a1}-{a2}', 'perSNPdiff_z'): diff_z,
                (f'{a1}-{a2}', 'perSNPdiff_p'): diff_p
            }
            result_dict.update(update_dict)

        return pd.Series(result_dict)


def info_pre():
    a = datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
    return f'INFO  {a} - '


if __name__ == '__main__':

    args = parser.parse_args()
    print('''
**************************************
* Testing h2g enrichment differences *
* (C) 2021 Lin Miao @ Miaoxin Li Lab *
* MIT License                        *
**************************************
''')
    annot_chr = args.annot_chr.split(',')
    if args.enrich_set1 and args.enrich_set2:
        enrich_set1 = args.enrich_set1.split(',')
        enrich_set2 = args.enrich_set2.split(',')
        enrich_both = enrich_set1 + enrich_set2
        if len(enrich_set1) == len(enrich_set2):
            print(f'\nEnrichments of the following pairs will be compared:')
            for s1, s2 in zip(enrich_set1, enrich_set2):
                print(f'{s1} vs {s2}')
            print()
        else:
            sys.exit('\nThe numbers of annotations in ENRICH_SET1 and ENRICH_SET2 should be same.\n')
    else:
        sys.exit('\nBoth of ENRICH_SET1 and ENRICH_SET2 are required.\n')

    print(f'{info_pre()}Read annotations.')
    stdev = read_stdev()
    M, M_5_50, M_Enrich = read_annotate()

    print(f'{info_pre()}Read LDSC result files.')
    results = OneTrait(args.ldsc_out)

    print(f'{info_pre()}Calculate tau stars.')
    tau_stars = results.cal_tau_star()

    print(f'{info_pre()}Compare enrichment differences.')
    enrichment = results.cal_enr_diff()

    df_results = pd.concat([tau_stars, enrichment])
    df_results.name = args.header if args.header else os.path.basename(args.ldsc_out)
    out_file = args.out if args.out else f'{args.ldsc_out}.tsv'
    df_results.to_csv(out_file, sep='\t')
    print(f'{info_pre()}Result wrote to {out_file}\n')
