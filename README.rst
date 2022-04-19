===================
ldsc_enrich_diff.py
===================

This script tests heritability enrichment differences between two sets of genomic regions with LDSC outputs.

When running LDSC, the flags of --overlap-annot, --print-coefficients, and --print-delete-vals are needed.



===================
mesc_enrich_diff.py
===================

This script tests expression-mediated heritability enrichment difference between two sets of genes with MESC outputs.

To use this script, the MESC script ``mesc/sumstats.py`` should be modified to output regression coefficients.
The objects of ``hsqhat.coef`` and ``hsqhat.part_delete_values`` are ndarrays of the whole data coefficients and the jackknife coefficients respectively.
Add the following lines to output them with suffixes of ``.GENE_SET1.coef_wholedata``, ``.GENE_SET1.coef_jackknife``, ``.GENE_SET2.coef_wholedata`` and ``.GENE_SET2.coef_jackknife``:

.. code-block:: python

    np.savetxt('MESC_OUT_PREFIX.GENE_SETx.coef_jackknife', hsqhat.part_delete_values, comments='')
    np.savetxt('MESC_OUT_PREFIX.GENE_SETx.coef_wholedata', hsqhat.coef, comments='')