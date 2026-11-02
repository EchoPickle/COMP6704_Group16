# 数学推导与数值约定

## 1. 从原始问题到对偶问题

给定 `n` 个样本，`x_i ∈ R^d`，`y_i ∈ {-1,+1}`，软间隔 SVM 求解：

\[
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_i\xi_i,
\quad 1-\xi_i-y_i(w^Tx_i+b)\le0,\quad-\xi_i\le0.
\]

令不等式乘子为 `alpha_i ≥ 0`、`mu_i ≥ 0`，拉格朗日函数为：

\[
\mathcal L=\frac12\|w\|^2+C\sum_i\xi_i+
\sum_i\alpha_i[1-\xi_i-y_i(w^Tx_i+b)]-\sum_i\mu_i\xi_i.
\]

分别令关于 `w,b,xi` 的导数为零：

\[
w=\sum_i\alpha_i y_i x_i,\qquad
\sum_i\alpha_i y_i=0,\qquad C-\alpha_i-\mu_i=0.
\]

由于 `mu_i ≥ 0`，有 `0 ≤ alpha_i ≤ C`。代回得到最大化对偶：

\[
D(\alpha)=\mathbf1^T\alpha-\frac12\alpha^TQ\alpha,
\quad Q_{ij}=y_i y_jK_{ij},\quad K_{ij}=x_i^Tx_j,
\]
\[
0\le\alpha_i\le C,\qquad y^T\alpha=0.
\]

`Q` 半正定，故目标凹；可行的 KKT 点是全局最优点。原始问题有严格可行点，满足强对偶。重复样本等情形下 alpha 可能非唯一，不能要求两个优化器返回完全相同的 alpha 数组。

## 2. KKT 与 margin 的三种状态

互补松弛为：

\[
\alpha_i[1-\xi_i-y_if_i]=0,\qquad(C-\alpha_i)\xi_i=0.
\]

设 `m_i = y_i f_i`：

| alpha 状态 | KKT margin 要求 | 违例残差 r_i |
|---|---|---|
| `alpha_i = 0` | `m_i ≥ 1` | `max(0,1-m_i)` |
| `0 < alpha_i < C` | `m_i = 1` | `abs(m_i-1)` |
| `alpha_i = C` | `m_i ≤ 1` | `max(0,m_i-1)` |

实现用 `bound_tol = min(1e-10, C*1e-8)` 判断 alpha 是否在边界附近；这是 alpha 的数值容差，与 margin 容差 `tol` 不同。很小的 C 下也不会把整个可行区间都当成边界。

每次扫描结束，用重新计算的全体 margin 检查 `max(r_i) ≤ tol`。同时记录 `abs(y @ alpha)` 和盒约束越界量。KKT 残差依赖数值边界分类；在极端病态输入下应同时审阅目标间隙与可行性。

## 3. 成对约束及可行区间

只允许改变 `alpha_i,alpha_j`，则：

\[
y_i\alpha_i'+y_j\alpha_j'=y_i\alpha_i+y_j\alpha_j.
\]

令 `s=y_i y_j`、`t=alpha_j'-alpha_j`，得到：

\[
\alpha_i'=\alpha_i-st,\qquad\alpha_j'=\alpha_j+t.
\]

将两个变量的盒约束同时施加到 `alpha_j'`：

\[
\begin{cases}
L=\max(0,\alpha_j-\alpha_i),\\
H=\min(C,C+\alpha_j-\alpha_i),
\end{cases}\quad y_i\ne y_j;
\]

\[
\begin{cases}
L=\max(0,\alpha_i+\alpha_j-C),\\
H=\min(C,\alpha_i+\alpha_j),
\end{cases}\quad y_i=y_j.
\]

若 `H ≤ L`，该对无可行移动。更新后由等式关系求伙伴变量；不可独立裁剪两个 alpha，否则会破坏等式约束。代码仅在舍入造成极小越界时同时修正伙伴。

## 4. 一维目标及解析更新

记 `E_i=f_i-y_i`，将成对变化代入二次目标，得到**精确**的目标增益：

\[
D(\alpha')-D(\alpha)=p t-\frac12\eta t^2,
\quad p=y_j(E_i-E_j),
\]
\[
\eta=K_{ii}+K_{jj}-2K_{ij}=\|x_i-x_j\|^2\ge0.
\]

`b` 从 `E_i-E_j` 中抵消。`eta>0` 时：

\[
t^*=p/\eta,\qquad
\alpha_j'=\operatorname{clip}(\alpha_j+t^*,L,H).
\]

实现直接用 `dot(x_i-x_j,x_i-x_j)` 算 eta，以减少三个大 Gram 元素相减的抵消误差；先将 `p` 与 `eta*(L-alpha_j)`、`eta*(H-alpha_j)` 比较，再决定是否需要除法，防止极小正曲率产生无意义的溢出。

`eta=0` 时目标沿可行线段是线性的，比较两个端点的 `p*t`，只接受正增益；若完全平坦则保持原值。这使标签冲突的重复样本也可正常优化。

误差差值如果不超过 `16*机器精度*max(1,abs(E_i),abs(E_j))`，视作没有可靠搜索方向，继续尝试其他伙伴，避免舍入噪声导致自由变量之间反复进行假更新。步长小于自身浮点精度尺度时同样拒绝，且要求解析增益严格为正。没有用 `tol` 充当步长阈值：margin 容差与 alpha 步长是不同量纲。浮点运算中记录的对偶目标可能出现机器精度量级抖动，测试允许 `1e-9` 的绝对数值误差。

## 5. 偏置与误差缓存

令 `delta_i=alpha_i'-alpha_i`、`delta_j=alpha_j'-alpha_j`。若 i 是自由支持向量，要求更新后的 `f_i'=y_i`，故：

\[
b_i=b-E_i-y_i\delta_iK_{ii}-y_j\delta_jK_{ij}.
\]

同理：

\[
b_j=b-E_j-y_i\delta_iK_{ij}-y_j\delta_jK_{jj}.
\]

优先使用自由 i 对应的 `b_i`，其次自由 j 的 `b_j`；均在边界时取两者均值。整个误差向量增量更新为：

\[
E_k'=E_k+y_i\delta_iK_{ki}+y_j\delta_jK_{kj}+(b'-b).
\]

每次扫描结束重新计算 `w`、分数与误差，消除缓存累积漂移。

### 用全体 KKT 条件校准 b

令 `g_i=w^T x_i`，`t_i=y_i-g_i`。定义：

\[
\mathcal L_b=\{i:y_i=+1,\alpha_i<C\}\cup\{i:y_i=-1,\alpha_i>0\},
\]
\[
\mathcal U_b=\{i:y_i=-1,\alpha_i<C\}\cup\{i:y_i=+1,\alpha_i>0\}.
\]

KKT 要求 `b ≥ max(t_i, i∈L_b)` 且 `b ≤ min(t_i, i∈U_b)`。自由变量会同时贡献上下界。记这些界为 `b_low,b_high`：

- `b_low ≤ b_high` 时，区间中任意 b 满足 margin-KKT；使用中点。
- 区间反向时，迭代还未最优；中点最小化固定 alpha 下最坏的 margin-KKT 违例，最小值为 `(b_low-b_high)/2`。

该方法不改变 alpha 或对偶目标，能处理没有自由支持向量的情况；不能仅对空自由集合求偏置均值。

## 6. 选择策略与停止

从可行初值 `alpha=0` 开始。外层首先扫描全部样本，然后扫描非边界样本；非边界扫描没有更新后回到完整扫描，并且连续非边界扫描最多五次。这个上限保证即使自由集合持续出现极小更新，也不会让仍违例的边界变量长期得不到检查。

对违反 KKT 的 i，先尝试自由变量 j，按 `abs(E_i-E_j)` 从大到小排序；再尝试全部边界 j。该误差差值是步长启发式，不保证最大目标增益，因此保留完整回退。先用固定种子随机打乱，再稳定排序，以复现平局处理。

每次扫描结束都检查**全部**样本，即使本轮只遍历了自由变量：

1. `max_kkt_violation ≤ tol`：成功收敛。
2. 完整扫描无更新且仍违例：数值停滞，警告，不能宣称收敛。
3. 扫描次数达到 `max_iter`：警告并保留当前可行解。

`n_iter_` 是扫描次数；`n_pair_updates_` 是成功成对更新次数；两者含义不同。原始目标或 KKT 残差不要求逐次单调；每次有效更新增加的是**对偶**目标。

## 7. 目标与间隙

给定当前 w,b，取最小可行松弛 `xi_i=max(0,1-y_i f_i)`：

\[
P=\tfrac12\|w\|^2+C\sum_i\max(0,1-y_if_i),\quad
D=\sum_i\alpha_i-\tfrac12\|w\|^2.
\]

若 alpha 对偶可行且 `w=X^T(alpha*y)`，则 `P-D ≥ 0`，并可写成：

\[
P-D=\sum_i\left[C\max(0,1-m_i)+\alpha_i(m_i-1)\right].
\]

精确边界分类下，若每个 KKT 违例 ≤ tol，则间隙 ≤ `n*C*tol`。代码保留原始间隙，不将负值强行截成零，以暴露浮点问题。间隙是额外诊断，本版本的停止准则是全体 margin-KKT。

## 8. 参考与独立实现范围

数学背景参考 [Platt 的 SMO 技术报告（1998）](https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf)。本项目根据上述约束和二次函数独立编写，未复制现成 SMO 源码。

外部对照使用 [SVC 官方接口](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)。它的迭代定义和停止容差与这里不完全相同，因此比较目标值、可行性、w,b、分数与预测，不把迭代次数当作同一单位。
