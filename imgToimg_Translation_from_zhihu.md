# Image-to-Image Translation

---

## GAN

**用分布来描述生成**

$P_{data}(x)$描述图像，${x_1, x_2, ..., x_m}$为sample，在$P_{data}$未知的情况下，用已知的分布$P_{G}(x;\theta)$去逼近，求$\theta$

$$
\begin{array}{l}
\theta^{*}=\arg \max _{\theta} \prod_{i=1}^{m} P_{G}\left(x^{i} ; \theta\right)=\arg \max _{\theta} \sum_{i=1}^{m} \log \left[P_{G}\left(x^{i} ; \theta\right)\right] \\
\approx \arg \max _{\theta} \mathbb{E}_{x \sim P_{\text {data }}} \log \left[P_{G}\left(x^{i} ; \theta\right)\right]
\end{array}

$$

将期望展开，加入一项化简

$$
\begin{array}{l}
\theta^{*}=\arg \max _{\theta} \int_{x} P_{\text {data }}(x) \log P_{G}\left(x^{i} ; \theta\right) d x-\int_{x} P_{\text {data }}(x) \log P_{\text {data }}(x) d x \\
=\arg \min _{\theta} K L\left(P_{\text {data }}(x) \| P_{G}\left(x^{i} ; \theta\right)\right)
\end{array}

$$

KL散度衡量两个分布，越小就越逼近，GAN中的discriminator要做的就是计算生成分布与真实分布的与KL divergence相关的一个散度。

现实生活中的分布很复杂，但是NN可以解决复杂分布，输入分布，NN也能输出分布。

用NN把一个简单的distribution映射成复杂的distribution后，无法用MLE计算复杂distribution的参数，也就是无法用KL divergence估计两个distribution的差异。

GAN提出的目的就是为了找出一种新的衡量distribution差异的方法。基本思路就是：

$$
G^{*}=\arg \min _{G} \max _{D} V(G, D) 

$$

其中， V(G, D) 称为value function(价值函数) 

$$
V(G, D)=\mathbb{E}_{x \sim P_{\text {data }}}[\log D(x)]+\mathbb{E}_{x \sim P_{G}}[\log (1-D(x))]

$$

给定G，最好的D为

$$
D^{*}=\frac{p_{\text {data }}(x)}{p_{\text {duta }}(x)+p_{G}(x)}

$$

给定不同的G可以算出不同的D，将D带回后

$$
\begin{array}{l}
V(G, D)=E_{x \sim P_{\text {data }}}[\log D(x)]+E_{x \sim P_{G}}[\log (1-D(x))] \\
=E_{x \sim p_{\text {data }}}\left[\log \frac{p_{\text {data }}(x)}{p_{\text {data }}(x)+p_{G}(x)}\right]+E_{x \sim p_{G}}\left[\log \left(1-\frac{p_{\text {data }}(x)}{p_{\text {data }}(x)+p_{G}(x)}\right)\right] \\
=E_{x \sim P_{\text {data }}}\left[\log \frac{p_{\text {data }}(x)}{p_{\text {data }}(x)+p_{G}(x)}\right]+E_{x \sim P_{G}}\left[\log \frac{p_{G}(x)}{p_{\text {data }}(x)+p_{G}(x)}\right] \\
=\int_{x} p_{\text {data }}(x)\left[\log \frac{\left[p_{\text {data }}(x)\right.}{p_{\text {data }}(x)+p_{G}(x)}\right]+p_{G}(x)\left[\log \frac{p_{G}(x)}{p_{\text {data }}(x)+p_{G}(x)}\right] d x \\
=-2 \log 2+\int_{x} p_{\text {data }}(x)\left[\log \frac{p_{\text {data }}(x)}{\left(p_{\text {data }}(x)+p_{G}(x)\right) / 2}\right]+p_{G}(x)\left[\log \frac{p_{G}(x)}{\left(p_{\text {data }}(x)+p_{G}(x)\right) / 2}\right] d x \\
=-2 \log 2+K L\left(P_{\text {data }}(x) \| \frac{P_{\text {data }}(x)+P_{G}(x)}{2}\right)+K L\left(P_{G}(x) \| \frac{P_{\text {data }}(x)+P_{G}(x)}{2}\right) \\
=-2 \log 2+2 J S D\left(P_{\text {data }}(x) \| P_{G}(x)\right)\\
\end{array}

$$

问题转化为

<img src="https://raw.githubusercontent.com/CalcuLuUus/pics/main/20220916165007.png"/>

最优化问题梯度下降，定一个求一个

<img src="https://raw.githubusercontent.com/CalcuLuUus/pics/main/20220916165136.png"/>

详情见[[GAN zhihu](https://zhuanlan.zhihu.com/p/34560149)]

---

## DCGAN

GAN的G和D用的是简单的MLE，替换成CNN

网络结构如下

<img src="https://raw.githubusercontent.com/CalcuLuUus/pics/main/20220916165312.png"/>
