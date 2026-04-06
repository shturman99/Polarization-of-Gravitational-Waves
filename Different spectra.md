Based on the BK2016 we can have different scaling laws for the length scale.

$$
\xi_{i}(t) = \int_{0}^{\infty} k^{-1}E_{i}(t,k)dk  \bigg/ \int_{0}^{\infty} E_{i}(t,k)dk
$$
$i$  here stands for two different cases Kinetic and Magnetic energy files.  We Introduce the same  theoretical modeling of the sealing:
$$
E(k \xi(t),t) =  \xi^{-\beta} \phi(k\xi)
$$
$\phi$ is function that determines the scaling based on the nature of the process, additionally we can assume $\xi \propto t^{q}$ with this assumption we can immediately get form  numerical analysis get that energy scales as $E\propto k^{\alpha}l^{-\alpha}$  where $\alpha=-3 + \frac{2}{q}$  or $q = \frac{2}{(\alpha+3)}$ ,(the parallelization with $\alpha$ was due to Olsen), same thing would be achieved with $\beta$, 
$$
\beta = \frac{2}{q} -3
$$
the $\mathcal{E}_{i}= \int_{0}^{\infty}E_{i}(t,k)dk\propto t^{-p_{i}}$   this is obtained by integrating over k so we  have relationship $p=(1+\beta)q$

And we want to study following 3 cases:

| $\beta$ | q   | p    |
| ------- | --- | ---- |
| 4       | 2/7 | 10/7 |
| 1       | 1/2 | 1    |
| 0       | 2/3 | 2/3  |

## Cases
Now we focus our attention on explicit functional forms of $E$ and its peick 
For $\beta=0$ 
We would have $k_{0}(t)=k_{0}(t_{0}) \left( 1+ \frac{t}{t_{1}} \right)^{-2/3}$
For $\beta=1$
We would have $k_{0}(t)= k_{0}(t_{0})(1+\frac{t}{t_{1}})^{-1/2}$
$$
E(k,t)=
\begin{cases}
\mathrm{Const}~ k^{s} ~~ k<k_{0}(t) \\
~ \\
\mathrm{Const'}~ k^{-5/3} ~~ k_{0}(t)<k
\end{cases}
$$
where $s=[2,4]$ one of this to cases
