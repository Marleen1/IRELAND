# IRELAND

# Iterative Rule Extension for Logic Analysis of Data (IRELAND): a MILP-based heuristic to derive interpretable binary classifiers from large datasets

With the increasing popularity of machine learning and data science, there is also an increasing need for trust in these models. One of the requirements for trust in machine learning models is the extent to which we understand them, i.e. the interpretability of the models and their outcomes. Boolean rules in disjunctive normal form (DNF) are an interpretable way of describing input-output relationships: they form an OR combination of AND clauses for classification. For example, a Boolean rule in DNF could look as follows: IF a sample has (feature X AND feature y) OR (feature A AND feature B AND feature C) OR (feature Q), THEN they are classified as case, else as control.

Extracting Boolean rules in DNF from large binary data is not a trivial task. IRELAND provides a heuristic that can extract these rules from data with up to 10,000 features and samples. The general idea behind IRELAND is to generate a large pool of AND clauses using a sub problem, from which a master problem can select the combination of AND clauses that leads to the best classification performance. This pool is constructed in such a way that it can also be used to generate a sensitivity-specificity trade-off curve. For more details on the algorithm, see [1].

When using this work in further publications, please cite [1].

What comes next in this README:
1. Which scripts can you find in this Github repo
2. Which datasets can you find in this Github repo
3. How to run the code
4. References
---

1. Which scripts can you find in this Github repo

There are four scripts available.
IRELAND.py: runs the original IRELAND algorithm as shown in Figure 1 in [1].
IRELAND_nonoise.py: the algorithm was tested on datasets that do not contain any noise. Shortly, this means that the parameter UB is set to 0, making the algorithm simpler.
IRELAND_featureSubsets.py: to speed up the algorithm, experiments were performed where the sub problem was solved while including only a subset of all the features. The results in [1] show that this does not lead to improvements.
BRCG.py: in [1], IRELAND was compared to the algorithm BRCG [2]. For a fair comparison between the two algorithms, an implementation of BRCG was necessary using the same assumptions and optimization algorithm as for IRELAND.
---

2. Which datasets can you find in this Github repo

In [1], four data collections are described. The synthetic datasets without noise and with noise can be found in nonoise_datasets.tar.gz and noise_datasets.tar.gz, respectively. The dataset where the genomes were obtained from the 1000 Genomes project with classes based on simulated DNF rules can be found in 1000Genomes_DNF.tar.gz. 1000Genomes_PEPS contains dataset datasets derived from the data made available by [3]. The genomes in the 1000Genomes were originally collected by [4].
---

3. How to run the code

To be able to run the script, the following programs need to be installed:
- Python 3
- Gurobi (through gurobipy)
- numpy

Each script can be run from the command line, where the scriptname is followed by the required input parameters. For example:

python IRELAND.py Inst444_N2500_P100_K5_M10_error0p025_rules Data/noise_datasets Results/noise_datasets/Inst444_N2500_P100_K5_M10_error0p025_rules Runfiles 4 3600 100 3 5

runs IRELAND for the dataset Inst444_N2500_P100_K5_M10_error0p025_rules, the data can be found in folder Data/noise_datasets, the results will be stored in Results/noise_datasets/Inst444_N2500_P100_K5_M10_error0p025_rules, files that IRELAND generates on the fly for running are stored in the folder Runfiles, Gurobi is allowed to use at most 4 threads, the script terminates after at most 3600 seconds, N_s = 100 (see [1] for details), the maximum number of clauses selected is 3 (K=3), and each clause can contain at most 5 features (M=5).

The four scripts require the following inputs, in the order provided.
IRELAND.py:
- dataset name;
- folder where the input data is located;
- folder where the results will be stored;
- folder where runfiles will be stored;
- maximum number of threads that Gurobi is allowed to use for each sub problem;
- maximum runtime of the algorithm;
- N_s size of the random subset of samples included in each iteration of the sub problem (see [1] for details);
- K (maximum number of clauses selected by the master problem);
- M (maximum number of features in a clause).

IRELAND_nonoise.py:
- dataset name;
- folder where the input data is located;
- folder where the results will be stored;
- maximum number of threads that Gurobi is allowed to use for each sub problem;
- maximum runtime of the algorithm;
- N_s size of the random subset of samples included in each iteration of the sub problem (see [1] for details);
- K (maximum number of clauses selected by the master problem);
- M (maximum number of features in a clause).

IRELAND_featureSubset.py:
- dataset name;
- folder where the input data is located;
- folder where the results will be stored;
- folder where runfiles will be stored;
- maximum number of threads that Gurobi is allowed to use for each sub problem;
- maximum runtime of the algorithm;
- N_s size of the random subset of samples included in each iteration of the sub problem (see [1] for details);
- P_s: size of the random subset of features included in each iteration of the sub problem (see [1] for details):
- K (maximum number of clauses selected by the master problem);
- M (maximum number of features in a clause).

BRCG.py:
- dataset name;
- folder where the input data is located;
- folder where the results will be stored;
- maximum number of threads that Gurobi is allowed to use for each sub problem;
- maximum runtime for each individual master- and sub problem;
- maximum runtime of the algorithm;
- K (maximum number of clauses selected by the master problem);
- M (maximum number of features in a clause).
---

4. References

[1] Balvert, M (2023) Iterative Rule Extension for Logic Analysis of Data (IRELAND): a MILP-based heuristic to derive interpretable binary classifiers from large datasets.
[2] Dash S, Gunluk O, Wei D (2018) Boolean decision rules via column generation. Advances in Neural Information Processing Systems, 4655–4665.
[3] Bayat A, Piotr S, O’Brian AR, Dunne R, Hosking B, Jain Y, Hosking C, Luo OJ, Twine N, Bauer DC (2020) Variantspark: Cloud-based machine learning for association study of complex phenotype and large-scale genomic data. GigaScience Database 9(8).
[4] 1000 Genomes Project Consortium (2015) A global reference for human genetic variation. Nature 526.
