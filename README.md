# Introduction

MIP benchmarks for a beer game (Multi-echelon inventory management problem with four facilities) using Gurobi.

## Install & Run

```shell
# Clone repo
$ git clone git@github.com:qihuazhong/multi-echelon-gurobi.git

# Install python dependencies
$ pip install -r requirements.txt
# Install multi-echelon-inventory environment
$ git clone -b dev-single-agent git@github.com:qihuazhong/multi-echelon-drl.git 

# Run
$ python main.py
```

## MIP formulation

$$
\begin{aligned}
\min \sum_{k\in K}\sum_{j\in J} &(C_{b}^{j}B_{k}^{j} + C_{h}^{j}I_{k}^{j})\\ \\
\text{Subject to:}\\
I_{k-1}^{1}+E_{k-LS^{1}}^{2}-B_{k-1}^{1} &=d_{k}+I_{k}^{1}-B_{k}^{1},\forall k\in K\\
I_{k-1}^{j}+E_{k-LS^{j}}^{j+1} &=E_{k}^{j}+I_{k}^{j},\forall k\in K,j\in J/\{1\}\\
\sum_{t=1}^{k}E_{t}^{j} &=\sum_{t=1}^{k}P_{t-LI^{j-1}}^{j-1}-B_{k}^{j}-B_{0}^{j},\forall k\in K,j\in J/\{1\}\\
B_{k}^{j},P_{k}^{j},I_{k}^{1} &\geq0,\forall j\in J,k\in K\\
E_{k}^{j} & \geq0,\forall j\in J/\{1\},k\in K\\
I^{j}_{k} &\leq x^{j}_{k}M, \forall j \in J, k \in {K}\\
B^{j}_{k} &\leq y^{j}_{k}M, \forall j \in J, k \in {K}\\
x^{j}_{k} + y^{j}_{k} &\leq 1, \forall j \in J, k \in {K}\\
x^{j}_{k}, y^{j}_{k} &\in \{0, 1\} \forall j \in J, k \in {K}\\
I^{j}_{k}, B^{j}_{k}, P^{j}_{k}, E^{j}_{k} &\geq 0, \forall j \in J, k \in K
\end{aligned}
$$

## Results sample

| Instance | Cost    | iterative_MIP | once_MIP   |
| -------- | ------- | ------------- | ---------- |
| 0        | -206.00 | -4,209.25     | -11,474.25 |
| 1        | -206.00 | -3,770.00     | -1,718.50  |
| 2        | -189.50 | -4,214.00     | -12,394.75 |
| 3        | -452.00 | -2,933.25     | -1,861.50  |
| 4        | -247.25 | -4,112.75     | -15,235.75 |
| 5        | -276.50 | -4,262.00     | -10,819.75 |
| 6        | -514.00 | -5,713.50     | -22,833.25 |
| 7        | -276.50 | -4,015.00     | -1,894.00  |
| 8        | -801.50 | -4,672.75     | -5,959.50  |
| 9        | -203.25 | -3,836.00     | -1,176.50  |
