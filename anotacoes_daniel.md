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

### TBM

## 3.4 THE TRIPLE-BARRIER METHOD
Here I will introduce an alternative labeling method that I have not found in the literature. If you are an investment professional, I think you will agree that it makes more sense. I call it the triple-barrier method because it labels an observation according to the first barrier touched out of three barriers. First, we set two horizontal barriers and one vertical barrier. The two horizontal barriers are defined by profit-taking and stoploss limits, which are a dynamic function of estimated volatility (whether realized or implied). The third barrier is defined in terms of number of bars elapsed since the position was taken (an expiration limit). If the upper barrier is touched first, we label the observation as a 1. If the lower barrier is touched first, we label the observation as a −1. If the vertical barrier is touched first, we have two choices: the sign of the return, or a 0. I personally prefer the former as a matter of realizing a profit or loss within limits, but you should explore whether a 0 works better in your particular problems.
You may have noticed that the triple-barrier method is path-dependent. In order to label an observation, we must take into account the entire path spanning [ti,0, ti,0 + h], where h defines the vertical barrier (the expiration limit). We will denote ti,1 the time of the first barrier touch, and the return associated with the observed feature is rti,0,ti,1 . For the sake of clarity, ti,1 ≤ ti,0 + h and the horizontal barriers are not necessarily symmetric.
Snippet 3.2 implements the triple-barrier method. The function receives four arguments:
- `close`: A pandas series of prices.
- `events`: A pandas dataframe, with columns,
    - `t1`: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
    - `trgt`: The unit width of the horizontal barriers.
- `ptSl`: A list of two non-negative float values:
    - `ptSl[0]`: The factor that multiplies trgt to set the width of the upper barrier. If 0, there will not be an upper barrier.
    - `ptSl[1]`: The factor that multiplies trgt to set the width of the lower barrier. If 0, there will not be a lower barrier.
- `molecule`: A list with the subset of event indices that will be processed by a single thread. Its use will become clear later on in the chapter.
SNIPPET 3.2 TRIPLE-BARRIER LABELING METHOD
```python
def applyPtSlOnT1(close,events,ptSl,molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    
    if ptSl[0]>0:pt=ptSl[0]*events_['trgt']
    else:pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0:sl=-ptSl[1]*events_['trgt']
    else:sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    return out
```
The output from this function is a pandas dataframe containing the timestamps (if
any) at which each barrier was touched. As you can see from the previous description,
the method considers the possibility that each of the three barriers may be disabled.
Let us denote a barrier configuration by the triplet [pt,sl,t1], where a 0 means
that the barrier is inactive and a 1 means that the barrier is active. The possible eight
configurations are:
 - Three useful configurations:
    - [1,1,1]: This is the standard setup, where we define three barrier exit conditions. We would like to realize a profit, but we have a maximum tolerance for
losses and a holding period.
    - [0,1,1]: In this setup, we would like to exit after a number of bars, unless we
are stopped-out.
    - [1,1,0]: Here we would like to take a profit as long as we are not stopped-out. This is somewhat unrealistic in that we are willing to hold the position for as
long as it takes.
- Three less realistic configurations:
    - [0,0,1]: This is equivalent to the fixed-time horizon method. It may still be useful when applied to volume-, dollar-, or information-driven bars, and multiple forecasts are updated within the horizon.
    - [1,0,1]: A position is held until a profit is made or the maximum holding
period is exceeded, without regard for the intermediate unrealized losses.
    - [1,0,0]: A position is held until a profit is made. It could mean being locked on a losing position for years.
- Two illogical configurations:
    - [0,1,0]: This is an aimless configuration, where we hold a position until we
are stopped-out.
    - [0,0,0]: There are no barriers. The position is locked forever, and no label is generated.

Figure 3.1 shows two alternative configurations of the triple-barrier method. On the left, the configuration is [1,1,0], where the first barrier touched is the lower horizontal one. On the right, the configuration is [1,1,1], where the first barrier touched is the vertical one.

## 3.5 LEARNING SIDE AND SIZE
In this section we will discuss how to label examples so that an ML algorithm can learn both the side and the size of a bet. We are interested in learning the side of a bet when we do not have an underlying model to set the sign of our position (long or short). Under such circumstance, we cannot differentiate between a profit-taking barrier and a stop-loss barrier, since that requires knowledge of the side. Learning the side implies that either there are no horizontal barriers or that the horizontal barriers must be symmetric.
Snippet 3.3 implements the function getEvents, which finds the time of the first
barrier touch. The function receives the following arguments:
 close: A pandas series of prices.
 tEvents: The pandas timeindex containing the timestamps that will seed every
triple barrier. These are the timestamps selected by the sampling procedures
discussed in Chapter 2, Section 2.5.
 ptSl: A non-negative float that sets the width of the two barriers. A 0 value
means that the respective horizontal barrier (profit taking and/or stop loss) will
be disabled.
 t1: A pandas series with the timestamps of the vertical barriers. We pass a
False when we want to disable vertical barriers.
 trgt: A pandas series of targets, expressed in terms of absolute returns.
 minRet: The minimum target return required for running a triple barrier search.
 numThreads: The number of threads concurrently used by the function.
SNIPPET 3.3 GETTING THE TIME OF FIRST TOUCH
def getEvents(close,tEvents,ptSl,trgt,minRet,numThreads,t1=False):
#1) get target
trgt=trgt.loc[tEvents]
trgt=trgt[trgt>minRet] # minRet
#2) get t1 (max holding period)
if t1 is False:t1=pd.Series(pd.NaT,index=tEvents)
#3) form events object, apply stop loss on t1
side_=pd.Series(1.,index=trgt.index)
events=pd.concat({'t1':t1,'trgt':trgt,'side':side_}, \
axis=1).dropna(subset=['trgt'])
df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index), \
numThreads=numThreads,close=close,events=events,ptSl=[ptSl,ptSl])
events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
events=events.drop('side',axis=1)
return events
Suppose that I = 1E6 and h = 1E3, then the number of conditions to evaluate
is up to one billion on a single instrument. Many ML tasks are computationally
LEARNING SIDE AND SIZE 49
expensive unless you are familiar with multi-threading, and this is one of them. Here
is where parallel computing comes into play. Chapter 20 discusses a few multiprocessing functions that we will use throughout the book.
Function mpPandasObj calls a multiprocessing engine, which is explained in
depth in Chapter 20. For the moment, you simply need to know that this function will
execute applyPtSlOnT1 in parallel. Function applyPtSlOnT1 returns the timestamps at which each barrier is touched (if any). Then, the time of the first touch is
the earliest time among the three returned by applyPtSlOnT1. Because we must
learn the side of the bet, we have passed ptSl=[ptSl,ptSl] as argument, and we
arbitrarily set the side to be always long (the horizontal barriers are symmetric, so
the side is irrelevant to determining the time of the first touch). The output from this
function is a pandas dataframe with columns:
 t1: The timestamp at which the first barrier is touched.
 trgt: The target that was used to generate the horizontal barriers.
Snippet 3.4 shows one way to define a vertical barrier. For each index in tEvents,
it finds the timestamp of the next price bar at or immediately after a number
of days numDays. This vertical barrier can be passed as optional argument t1
in getEvents.
SNIPPET 3.4 ADDING A VERTICAL BARRIER
t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
t1=t1[t1<close.shape[0]]
t1=pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]) # NaNs at end
Finally, we can label the observations using the getBins function defined in Snippet 3.5. The arguments are the events dataframe we just discussed, and the close
pandas series of prices. The output is a dataframe with columns:
 ret: The return realized at the time of the first touched barrier.
 bin: The label, {−1, 0, 1}, as a function of the sign of the outcome. The function
can be easily adjusted to label as 0 those events when the vertical barrier was
touched first, which we leave as an exercise.
SNIPPET 3.5 LABELING FOR SIDE AND SIZE
def getBins(events,close):
#1) prices aligned with events
events_=events.dropna(subset=['t1'])
px=events_.index.union(events_['t1'].values).drop_duplicates()
px=close.reindex(px,method='bfill')
50 LABELING
#2) create out object
out=pd.DataFrame(index=events_.index)
out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
out['bin']=np.sign(out['ret'])
return out