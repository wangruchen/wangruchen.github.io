---
layout: default
title: Multiple Kernel Learning Algorithms论文笔记
tags: [machine learning, MKL]
---

# {{ page.title }}



这篇文章第二部分主要讲MKL算法的关键性质，区分不同种算法间的异同。第三部分分类讨论不同MKL算法。第四部分实验。第五部分结论。

## 多核学习六个重要性质

了解这六个性质可以对MKL算法有一个清楚的分类。这六个性质：

1. the learning method
2. the functional form
3. the target function
4. the training method
5. the base learner
6. computation complexity

### The Learning Method

The existing MKL algorithms use different learning methods for determining the kernel combination function. 多核学习算法用不同的学习方法来决定核的结合函数。

主要有5类：

1. **Fixed rules**: don't need any training and parameters. 

	按照固定的形式结合核函数，结合函数不需要参数和训练。
	
2. **Heuridtic approaches**: use a parameterized combination function and find the parameters of this function generally by looking at some measure obtained from each kernel function separately.（分别计算出每个核函数对应的参数）

	该方法采用含有参数的结合函数。参数根据每个核函数的表现或核矩阵计算得到。

3. **Optimization approaches**: use a parametrized combination function and learn the parameters by solving an optimization problem.（通过解优化问题计算参数）

	该方法采用含有参数的结合函数。参数通过解优化问题得到。
	
	<!--这个优化问题可以是similarity measures或structural risk minimization approaches。-->

4. **Bayesian approches**: interpret the kernel combination parameters as random variables, put priors on these parameters, and perform inference for learning them and the base learner parameters.

	
5. **Boosting approaches**: inspired from ensemble and boosting methods, iteratively add a new kernel until the performance stops improving.

	灵感来自于集成学习。不断增加新的核知道效果不再提高。

### The Functional Form

核结合的函数形式可以分为下面3类：

1. **Linear combination method**: 

	线性结合方法：基本分为两类：无权重和（unweighted sum）、有权重和（weighted sum）。
	

2. **Nonlinear combination methods**: 

	非线性结合方法：用非线性函数，例如乘法、求幂等。

3. **Data-dependent combination method**: assign specific kernel weights for each data instance. 

	依赖数据的结合方法：为每个数据分配特定的核函数权重。

### The Target Function

通过优化不同的目标函数（target function）来得到结合函数的参数。目标函数分为三个基本类别：

1. **Similarity-based functions**: 

	基于相似度的函数：计算合并的核模型和训练集得到的最优核模型之间的相似度量，设置参数使得相似度最大。计算两个核之间相似度的方法有：kernel alignment, Euclidean distance, Kullback-Leibler (KL) divergence, or any other similarity measure。
<!--calculate a similarity metric between the combined kernel matrix and an optimum kernel matrix calculated from the training data and select the combination function parameters that maximize the similarity. -->
2. **Structural risk functions**: 

	结构风险函数：结构风险最小化，最小化正则项之和（对应于降低模型的复杂性和误差）。对核权重的约束可以结合正则项。For example, structural risk function can use the l1-norm, the l2-norm, or a mixed-norm on the kernel weights or feature spaces to pick the model parameters.


3. **Bayesian functions**: measure the quality of the resulting kernel function constructed from candidate kernels using a Bayesian formulation.

	贝叶斯函数：通常用likelihood或posterior作为目标函数，找到最大likelihood或最大posterior来确定参数。

### The Training Method

训练方法：

1. **One-step methods**: calculate both the combination function parameters and the parameters of the combined base learner in a single pass.

	不采用迭代，一次计算结合函数的参数和基础分类器参数。先计算出结合函数的参数，然后采用结合的核训练出一个分类器。

2. **Two-step methods**: use an iterative approach where each iteration, first we update the combination function parameters while fixing the base learner parameters, and then we update the base learner parameters while fixing the combination function parameters.

	采用迭代的方法，首先更新结合函数的权重参数，然后更每个基础分类器的参数，不断重复上面两步直到收敛。

### The Base Learner

目前有很多基于核的学习方法（kernel-based learning algorithm）：

- SVM和support vector regression (SVR)
- Kernel Fisher Discriminant analysis (KFD，核函数Fisher鉴别)：先对输入空间的样本进行非线性映射，变换到特征空间，然后在特征空间中利用线性Fisher鉴别寻找易于分类的投影线。
- Regularized kernel discriminant analysis (RKDA)
- kernel ridge regression (KRR，核岭回归) 

### The computational Complexity

多核学习方法的计算复杂度主要决定于两个方面：训练方法（是one-step还是two-step）和基本分类器的计算复杂程度。

## 3

### 3.9 Structural Risk Optimizing Nonlinear Approaches

Ong et al. (2003)提出学习一个核函数来代替核矩阵。在核空间定义一个核函数称为（Hyperkernels）。

Varma and Babu (2009)提出了generalized multiple kernel learning (GMKL)，它的目标函数中包含两个正则化项和一个损失函数。

<!--## 多核学习算法分类讨论

根据多核学习的性质，将现有的多核学习方法分为12类。

### Fixed Rules

在Pavlidis et al. (2001)的文章中，计算每一组数据的核，然后将它们加起来（无权重）。

Ben-Hur and Noble (2005)结合pairwise kernels，无权重相加。

### Heuristic Approaches


# Experiment

数据集：four dataset
核函数：linear kernel, gaussian kernel

## Compared algorithms

RBMKL
ABMKL
CABMKL
MKL
SimpleMKL
GMKL
GLMKL
NLMKL
LMKL
-->

## 其他

- Quadratic Programming (QP，二次规划)：如果目标函数为凸二次函数，这个优化问题成为二次优化问题：![QP](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/QP.png)
- (SOCP，二次锥规划问题)
- Quadratically Constrained Quadratic Programming (QCQP，二次约束的二次规划)：如果目标函数和约束项是凸二次函数，称这个凸优化问题为二次约束二次规划问题：![QCQP](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/QCQP.png)
- Semidefinite Programming (SDP，半正定规划)：![SDP](file:///Users/wangruchen/work/learningMaterials/MachineLearning/MultipleKernelLearning/figure/SDP.png)
- Semi-infinite Linear Program (SILP，半无限线性规划)：一个优化问题，有有限的变量和无限的约束条件，或者有无限的变量和有限的约束条件。



## 总结

多核学习根据核函数结合方法的不同可以分为：

1. 无权重参数：fixed rules（按照特定的形式结合），Boosting approaches（像集成学习类似）
2. 有权重参数：Heuristic approaches，Optimization approaches（通过解优化问题得到参数），Bayesian approaches

得到结合参数时解的优化问题可以分为：

1. Similarity-based functions
2. Structural risk functions
3. Bayesian functions

按核函数的结合方式可以分为：

1. Linear combination（相加或求平均等）
2. Nonlinear combination（相乘或取幂等）
3. Data-dependent combination

训练过程有无迭代：

1. One-step methods
2. Two-step methods

基本学习方法：

1. SVM SVR
2. KFDA (Kernel Fisher discriminant analysis)
3. RKDA (Regularized kernel discrimi- nant analysis)
4. KRR (Kernel ridge regression)

