---
layout: default
title: SimpleMKL论文笔记
---

# {{ page.title }}


<script type="text/javascript"
 src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

(Optimize Linear risk 2-step)

这篇文章提出了SimpleMKL，一个新的MKL算法，基于mixed-norm regularization。SimpleMKL不仅可以被用于二分类，还可以被用在回归、聚类或多分类。

## Introduction

多核学习的问题是学习参数<img src="http://www.forkosh.com/mathtex.cgi? \alpha_{i}">和权重<img src="http://www.forkosh.com/mathtex.cgi? d_{m}">。<img src="http://www.forkosh.com/mathtex.cgi? \alpha_{i}">是拉格朗日系数，<img src="http://www.forkosh.com/mathtex.cgi? d_{m}">是每个核函数结合的权重。Lanckriet et al. (2004b)介绍了针对二分类的多核学习问题，随着样本的增加和核变大，约束的二次规划问题变得更困难（resulting in a quadratically constrained quadratic programming problem that becomes rapidly intractable as the number of learning examples or kernels become large）。造成这个的原因是它是一个凸的而不是平滑最小化的问题。


在这篇文章中针对MKL问题的另个一构想。用a weighted l2-norm regularization代替mixed-norm regularization。这个构想得到了一个平滑凸优化函数。

这篇文章的主要贡献是提出SimpleMKL，通过引入a weighted l2-norm regularization，解决MKL问题。这种方法的本质是基于梯度下降法。

文章第二部分介绍本文MKL的功能公式。第三部分是详细的算法描述和计算复杂度。第四部分讨论本文算法在其他SVM问题上的扩展。第五部分室实验结果的计算复杂度和其他方法的对比。

## Multiple Kernel Learning Framework

### Function Framework

$$
K(x,x')=\sum^{M}_{m=1}d_{m}K_{m}(x,x')
$$
MKL的目标就是在学习决策函数的过程中计算参数集合<img src="http://www.forkosh.com/mathtex.cgi? d_{m}">。（接下来是怎样解决这个问题）

### Multiple Kernel Learning Primal Problem

MKL中的决策函数：
$$
f(x)+b=\sum_{m}f_{m}(x)+b
$$
其中<img src="http://www.forkosh.com/mathtex.cgi? f_{m}">属于不同RKHS。

MKL的原始问题：
$$
\min_{f_{m},b,\xi,d}\frac{1}{2}\sum_{m}\frac{1}{d_{m}}||f_{m}||^{2}_{H_{m}}+C\sum_{i}\xi_{i}
$$
$$
s.t. y_{i}\sum_{m}f_{m}(x_{i})+y_{i}b\ge1-\xi_{i}
$$
$$
\xi_{i}\ge0
$$
$$
\sum_{m}d_{m}=1, d_{m}\ge0
$$

转换为拉格朗日问题：![拉格朗日问题](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/laSimpleMKL.png)

令上式的梯度等于零解得：![解](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/laSolveSimpleMKL.png)

## Algorithm for Solving the MKL Primal Problem

解MKL的原始问题。用MKL求最优分类超平面，它的原始优化问题为：![原始问题](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/SimpleMKLPrimal.png)

将原始问题转换为拉格朗日问题得到：![拉格朗日问题](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/laSimpleMKL.png)

上式要取得最大值，令其梯度等于零解得：![解](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/laSolveSimpleMKL.png)

### Computing the Optimal SVM Value and its Derivatives

将解得的结果再代入到拉格朗日式子中，转换为对偶问题得到：![对偶问题](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/dualSimpleMKL.png)

函数<img src="http://www.forkosh.com/mathtex.cgi? J(d)">被定义为对偶问题的目标值：![](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/JSimpleMKL.png)

这篇文章将解决上面问题转换为简化梯度法（reduced gradient method）

### Reduced Gradient Algorithm

<!--该算法迭代终止的条件有：duality gap, the KKT conditions, the variation of d between two consecutive steps or, even more simply, on a maximal number of iterations。这篇文章中采用的是duality gap（对偶间隙，指原始问题和对偶问题目标函数之间的差值）。-->

解上面这个问题通过reduced gradient method这个方法，先计算出J(d)的梯度和梯度方向：

![Jd](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/JdSimpleMKL.png)
![Dm](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/DmSimpleMKL.png)


需要沿梯度方向更新d（<img src="http://www.forkosh.com/mathtex.cgi? d \gets d+\gamma D">）；更新d需要找到在梯度方向上找最大的步长<img src="http://www.forkosh.com/mathtex.cgi? \gamma">，检查目标值J(d)是否下降，如果继续下降就更新d；迭代直到目标值J(d)停止下降，就可以得到最优的步长，同时得到最终的d。

<!--计算出梯度下降的方向：![梯度下降方向](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/desDirSimpleMKL.png)

然后找到梯度方向上的最大步长，检查此处的<img src="http://www.forkosh.com/mathtex.cgi? J(d)">是否下降。如果<img src="http://www.forkosh.com/mathtex.cgi? J(d)">下降，则更新d，令<img src="http://www.forkosh.com/mathtex.cgi? D_{v}=0">并归一化D。不断重复这个过程，直到<img src="http://www.forkosh.com/mathtex.cgi? J(d)">不再下降。在这一点，找到最优步长<img src="http://www.forkosh.com/mathtex.cgi? \gamma">.-->

## Conclusion

这篇文章提出了SimpleMKL模型，这种方法等同于其他MKL。但方法中主要增加了下降方法来解决优化问题。

# 其他

1. 选定核函数，以及每个特征采用的核函数。
2. 对训练集进行训练得到支持向量、每个支持向量对应的拉格朗日系数、每个核函数的权重。
3. 测试集特征，根据上面得到的支持向量、核函数权重，得到测试集的每个样本和支持向量间的核。
4. 根据得到的测试集的核和拉格朗日系数得到测试集的分类。

<!--### Connections With mixed-norm Regularization Formulation of MKL




**semi-infinite programming (SIP)**: an optimization problem with a finite number of variables and an infinite number of constraints, or an infinite number of variables and a finite number of constraints. 

**Reproducing Kernel Hilbert Space (RKHS，再生核希尔伯特空间)**: 内积空间，高维空间-->