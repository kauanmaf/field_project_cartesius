## Método do horizonte fixado

As it relates to finance, virtually all ML papers label observations using the fixed-time horizon method. This method can be described as follows. Consider a features matrix $X$ with $I$ rows, $\{X_i\}_{i=1,…,I}$, drawn from some bars with index $t = 1,…, T$, where $I ≤ T$. An observation $X_i$ is assigned a label $y_i \in {−1, 0, 1}$,

$$
    y_i = \begin{cases}
        -1 & \text{if} & r_{t_{i,0}, t_{i,0 + h}} < - \tau \\
        0 & \text{if} & |r_{t_{i,0}, t_{i,0 + h}}| \le \tau \\
        1 & \text{if} & r_{t_{i,0}, t_{i,0 + h}} > \tau
    \end{cases}
$$

Where $\tau$ is a pre-defined constant threshold, $t_{i,0}$ is the index of the bar immediately after $X_i$ takes place, $t_{i,0} + h$ is the index of the h-th bar after $t_{i,0}$, and $r_{t_{i,0}, t_{i,0 + h}}$ is the price return over a bar horizon h,

$$
    r_{t_{i,0}, t_{i,0 + h}} = \frac{p_{t_{i,0} + h}}{p_{t_{i,0}}} - 1
$$

## Qual fórmula de retorno usar
### Retorno logarítmico
Apresenta mais consistência estatisticamente, sendo menor sensível à outliers e skewness nos dados, além de permitir uma maior agregação de retorno ao longo do tempo (n entendi direito essa parte)

### Retorno aritmético

## Rotulagem com _rolling horizon_
- $r_{t_{i,0}, t} \ge \tau$: rotula como 1
- $r_{t_{i,0}, t} \le - \tau$: rotula como -1
- Se nenhum dos dois limiares for cruzado em uma horizonte de tempo máximo, rotulamos como 0.

Se o retorno em um período (dinâmico, menor que um $t$ máximo definido) for maior que $\tau$ ou menor que $- \tau$, rotulamos como 1 ou -1, respectivamente. Se nenhum dos limiares for alcançado antes do horizonte de tempo $t$, rotulamos como 0. Esse método é mais interessante porque permite um processo de rotulagem mais dinâmico e que captura melhor tendências importantes em dados voláteis.
(mais um método pra testar)

## O que fazer
Desenvolver rotulos sem preocupar com tamanho do investimento, apenas com a direção
Ta dando algum problema com o tipo do tau