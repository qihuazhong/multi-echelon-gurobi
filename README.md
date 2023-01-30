

# Introduction
MIP benchmark for a beer game (Multi-echelon inventory management problem with four facilities) with perfect information using Gurobi.

## Run
```shell
$ git clone -b dev-single-agent git@github.com:qihuazhong/multi-echelon-drl.git 
```


## MIP formulation
$$
\begin{aligned}
& \min \sum_{k\in K}\sum_{j\in J}(C_{b}^{j}B_{k}^{j} + C_{h}^{j}I_{k}^{j})\\ \\
I_{k-1}^{1}+E_{k-LS^{1}}^{2}-B_{k-1}^{1} &=d_{k}+I_{k}^{1}-B_{k}^{1},\forall k\in K\\
I_{k-1}^{j}+E_{k-LS^{j}}^{j+1} &=E_{k}^{j}+I_{k}^{j},\forall k\in K,j\in J/\{1\}\\
\sum_{t=1}^{k}E_{t}^{j} &=\sum_{t=1}^{k}P_{t-LI^{j-1}}^{j-1}-B_{k}^{j}-B_{0}^{j},\forall k\in K,j\in J/\{1\}\\
B_{k}^{j},P_{k}^{j},I_{k}^{1} &\geq0,\forall j\in J,k\in K\\
E_{k}^{j} & \geq0,\forall j\in J/\{1\},k\in K
\end{aligned}
$$

$$