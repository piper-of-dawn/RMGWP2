### Step 1 

##### a - pseudocode for the forward algorithm and the backward algorithm

##### Forward Algorithm

1. Let's initialize α matrix of size T x N with zeros
2. For each state i from 1 to N:
       α[0][i] = π[i] * b[i][o[0]]
3. For t from 1 to T-1:
       For each state i from 1 to N:
           α[t][i] = b[i][o[t]] * sum(α[t-1][j] * a[j][i] for j from 1 to N)
4. Probability of observing the sequence is sum(α[T-1][i] for i from 1 to N)

##### Toy example 


```python
# Consider a toy HMM with 2 states (A and B) and 3 observations (o1, o2, o3).
π = [0.6, 0.4]
a = [[0.7, 0.3], [0.4, 0.6]]
b = [[0.5, 0.5], [0.1, 0.9]]
o = [0, 1, 0]

# Initialize α matrix
α = [[0 for _ in range(2)] for _ in range(3)]
α[0][0] = π[0] * b[0][o[0]]
α[0][1] = π[1] * b[1][o[0]]

# Calculate α values
for t in range(1, 3):
    for i in range(2):
        α[t][i] = b[i][o[t]] * sum(α[t-1][j] * a[j][i] for j in range(2))

# Probability of observing the sequence
probability = sum(α[2][i] for i in range(2))
print("Probability of observing the sequence:", probability)

```

    Probability of observing the sequence: 0.06961600000000001
    

##### Backward Algorithm

1. Let's initialize β matrix of size T x N with zeros
2. For each state i from 1 to N:
       β[T-1][i] = 1
3. For t from T-2 to 0:
       For each state i from 1 to N:
           β[t][i] = sum(a[i][j] * b[j][o[t+1]] * β[t+1][j] for j from 1 to N)
4. Probability of observing the sequence is sum(π[i] * b[i][o[0]] * β[0][i] for i from 1 to N)

##### Toy Example


```python
π = [0.6, 0.4]
a = [[0.7, 0.3], [0.4, 0.6]]
b = [[0.5, 0.5], [0.1, 0.9]]
o = [0, 1, 0]

# Initialize β matrix
β = [[0 for _ in range(2)] for _ in range(3)]
for i in range(2):
    β[2][i] = 1

# Calculate β values
for t in range(1, -1, -1):
    for i in range(2):
        β[t][i] = sum(a[i][j] * b[j][o[t+1]] * β[t+1][j] for j in range(2))

# Probability of observing the sequence
probability = sum(π[i] * b[i][o[0]] * β[0][i] for i in range(2))
print("Probability of observing the sequence:", probability)

```

    Probability of observing the sequence: 0.069616
    

#### b - backward Viterbi Algorithm Pseudocode

1. Let's initialize δ matrix of size T x N with zeros
2. Lett's Initialize ψ matrix of size T x N with zeros
3. For each state i from 1 to N:
       δ[0][i] = π[i] * b[i][o[0]]
       ψ[0][i] = 0
4. For t from 1 to T-1:
       For each state i from 1 to N:
           δ[t][i] = max(δ[t-1][j] * a[j][i] for j from 1 to N) * b[i][o[t]]
           ψ[t][i] = argmax(δ[t-1][j] * a[j][i] for j from 1 to N)
5. Path[T-1] = argmax(δ[T-1][i] for i from 1 to N)
6. For t from T-2 to 0:
       Path[t] = ψ[t+1][Path[t+1]]


```python
π = [0.6, 0.4]
a = [[0.7, 0.3], [0.4, 0.6]]
b = [[0.5, 0.5], [0.1, 0.9]]
o = [0, 1, 0]

# Initialize δ and ψ matrices
δ = [[0 for _ in range(2)] for _ in range(3)]
ψ = [[0 for _ in range(2)] for _ in range(3)]
for i in range(2):
    δ[0][i] = π[i] * b[i][o[0]]
    ψ[0][i] = 0

# Calculate δ and ψ values
for t in range(1, 3):
    for i in range(2):
        δ[t][i] = max(δ[t-1][j] * a[j][i] for j in range(2)) * b[i][o[t]]
        ψ[t][i] = max(range(2), key=lambda j: δ[t-1][j] * a[j][i])

# Backtrack to find the most probable path
path = [0] * 3
path[2] = max(range(2), key=lambda i: δ[2][i])
for t in range(1, -1, -1):
    path[t] = ψ[t+1][path[t+1]]
print("Most probable path:", path)

```

    Most probable path: [0, 0, 0]
    

#### c - pseudocode for the Baum-Welch algorithm

1. Let's initialize π, a, and b randomly
2. Repeat until convergence:
3. Let's initialize ξ and γ matrices
4. For t from 0 to T-2:
        For each state i from 1 to N:
            For each state j from 1 to N:
                ξ[t][i][j] = (α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j]) / sum(α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j] for i,j from 1 to N)
5. For t from 0 to T-1:
        For each state i from 1 to N:
            γ[t][i] = sum(ξ[t][i][j] for j from 1 to N)
6. Update π, a, and b:
        π = γ[0]
        For each state i from 1 to N:
            For each state j from 1 to N:
                a[i][j] = sum(ξ[t][i][j] for t from 0 to T-2) / sum(γ[t][i] for t from 0 to T-2)
            For each observation k from 1 to M:
                b[i][k] = sum(γ[t][i] for t from 0 to T-1 if o[t] == k) / sum(γ[t][i] for t from 0 to T-1)


```python
π = [0.6, 0.4]
a = [[0.7, 0.3], [0.4, 0.6]]
b = [[0.5, 0.5], [0.1, 0.9]]
o = [0, 1, 0]

# Initialize randomly
import numpy as np
np.random.seed(42)

π = np.random.rand(2)
π /= π.sum()

a = np.random.rand(2, 2)
a /= a.sum(axis=1, keepdims=True)

b = np.random.rand(2, 2)
b /= b.sum(axis=1, keepdims=True)

# Baum-Welch Algorithm
converged = False
while not converged:
    ξ = np.zeros((3, 2, 2))
    γ = np.zeros((3, 2))

    # Calculate ξ and γ values
    for t in range(2):
        for i in range(2):
            for j in range(2):
                ξ[t][i][j] = (α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j]) / sum(α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j] for i in range(2) for j in range(2))

    for t in range(3):
        for i in range(2):
            γ[t][i] = sum(ξ[t][i][j] for j in range(2))

    # Update π, a, and b
    π = γ[0]
    for i in range(2):
        for j in range(2):
            a[i][j] = sum(ξ[t][i][j] for t in range(2)) / sum(γ[t][i] for t in range(2))

    for i in range(2):
        for k in range(2):
            b[i][k] = sum(γ[t][i] for t in range(3) if o[t] == k) / sum(γ[t][i] for t in range(3))
    
    # Check for convergence (simple version)
    converged = True

# Learned parameters
print("π:", π)
print("a:", a)
print("b:", b)

```

    π: [0.88669541 0.11330459]
    a: [[0.53380055 0.46619945]
     [0.23140517 0.76859483]]
    b: [[0.63742149 0.36257851]
     [0.18607042 0.81392958]]
    

### Step 2 

*a- Bull Regimes*

A bull regime in the Financial Market is marked as a time when there is an overall rise in the prices of stocks, accompanied by high investor confidence, optimism, and expectations of strong future performance along with some speculation. During the bull market, investors tend to buy more in the hope of expected payout leading to a sustained upward trend in the market.

Key Indicators: Increased IPOs of tech companies, and high trading volumes, low interest rates, and significant capital inflows into equity markets, Rise in Index levels.

Example of Bull Regimes:

Dot-com Bubble (2000):
As in the late 1900s, when the internet became mainstream leading to a massive surge in prices of technology stocks, and key driver of this growth was the wide adpatiation of the internet. Due to this sudden surge in the market, people started investing heavily mostly speculative investors, leading to the eventual market bubble.

Example: Amazon.com saw its stock price skyrocket from around $'1.50, in 1997 to over dollar 90 by the end of 1999.
Source: "The Dot-Com Bubble Burst" by Investopedia.

Post-Financial Crisis Recovery (2009-2020):
Post 08's Financial crisis, we saw the central bank implementing various monetary easing policies to help the market recover and restore investors' confidance in the Market. Which eventually lead to low intereste rate (availability of leverge) along with increase liquidity. Due to all these stimuli various asset classes showed upward trends leading to a bull market.
        
Example: Rise in S&P 500 index from a low of around 676 in March 2009 to over 3,200 by the end of 2019, marking one of the longest bull markets in history.
Source: "The Longest Bull Market in History: March 2009 – February 2020", by Nasdaq.

*b - Bear Regimes*

it is a period in the Financial Market when we see a downturn, ie continuous decline in prices across the various stocks, due to some external influence or global event impacting the market. This leads to decrease in fatih of investors in market leading to selling calls across the board. During this period market participants become skeptical about expected future returns.

Key Indicators: Rising unemployment rates, and widespread financial distress, Plummeting stock prices, bankruptcies of tech companies, and significant loss of investor wealth.

Example of Bear Regimes:

Global Financial Crisis (2007-2009):
This Crisis was man-made, it started in the real estate sector as sub subprime mortgage crisis, and due to heavy leverage eventually entered the equity market causing liquidity concerns, and exacerbating the impact of the decline in prices. This domino effect lead to significant drop in investors confidance in the overall market. 

Example: The Dow Jones Industrial Average fell from approximately 14,000 in October 2007 to a low as 6,600 in March 2009.
Source: "The Financial Crisis of 2007-08" by Federal Reserve History.

Dot-com Bust (2000-2002):
Every bull cycle is followed by a bear cycle, this was the case in early 2000, when after Dot-com bust, we saw a sharp decline in prices of technology stocks, as many internet companies failed to deliver the returns they promised, leading to mass exist and eventually price drops of tech stocks. 
 
Example: The NASDAQ Composite Index dropped from level of 5,000 in March 2000 to approx 1,100 by October 2002.
Source: "Dot-com Bubble Burst" by the U.S. Securities and Exchange Commission.


*c- Stagnant Regimes*

A period of confusion or flat market, is a period where we observe no significant movement in the Market, this is due to feelings of uncertainty in market participants. There can be various reason for that like related policy or prolonged global crises such as wars. In this period as participants are not sure which direction the market will swing, they tend to take precautions leading to low trading volume and stagnant prices.

Key Indicator: High inflation rates, and slow GDP growth, Flat stock market indices and deflation.

Examples of Stagnant Regimes:

1970s Stagflation Period:
It was a period of stalled economic growth, high inflation, and unemployment, commonly known as stagflation. This period marks the time when we saw little to no growth in the prices of stocks. This was a period of uncertainty and multiple global conflicts.
        
Example: S&P 500 Index fluctuated within a narrow range between 80 and 110 from 1973 to 1981.
Source: "Stagflation: Definition, Causes, and Examples" by The Balance.

Japanese Lost Decade (1990s):
This is one of the widely known examples, after a crash in Japan's asset market (From the 80s boom), the country experienced a decade of stagnation with almost little increase in asset prices with constant deflationary pressure. 

Example: The Nikkei 225 Index remained flat, revolving around 15,000 points for most of the 1990s, after its peak in 1989 at nearly 39,000.
Source: "The Lost Decade: Lessons From Japan’s Real Estate Crisis" by the IMF.


### Step 3

### Definition 3: Hidden Markov Model

A Hidden Markov Model is a quintuple $(Q, P, \Pi, A, B)$, where:

- $Q = \{q_1, q_2, \ldots, q_N\}$ is a finite set of $N$ hidden states.
- $P = \{s_1, s_2, \ldots, s_M\}$ is a finite set of $M$ observable symbols.
- $\Pi = \{\pi_i\}$ is initial probability distribution over these states.
- $A = \{a_{ij}\}$ is state transition probability matrix, where $a_{ij} = P(q_{t+1} = q_j \mid q_t = q_i)$.
- $B = \{b_i(s_k)\}$ is the emission probability matrix, where $b_i(s_k) = P(s_k \mid q_i)$.

It can be compactly represented as $\lambda = (\Pi, A, B)$ and must satisfy the following constraints:

1. The total probability of starting in any state is 1:

   $$
   \sum_{i=1}^{N} \pi_i = 1
   $$

2. The total transition probability from any state to all other states is 1 for each state $q_i$ in $Q$:

   $$
   \sum_{j=1}^{N} a_{ij} = 1 \quad \forall i \in Q
   $$

3. The total probability of emitting all possible symbols from any state $q_i$ is 1:

   $$
   \sum_{k=1}^{M} b_i(s_k) = 1 \quad \forall i \in Q
   $$

**Explanation:**

HMM is well know statistical model, which reprsents the systmes that undergos transitions between hidden states. Which lead to observable outups from each state. The states $Q$ represents configuration of the systm that are not observable directly. They can only be observed through sympbols from the set $P$, which are the measurable outputs.


- **Initial State Distribution ($\Pi$):** This is vector representation of probability of system starting in each hidden states. It is possbility of being in each state at initial time.
- **Transition Probability Matrix (A):** It is defined as the probabilities of moving from one state to another. $a_{ij}$ is the probability of transitioning from state $q_i$ to state $q_j$ in the next time step.
- **Emission Probability Matrix (B):** Emission Probability matrix is defnied as probability of observing each symbol given the current state. $b_i(s_k)$ is the probability of observing symbol $s_k$ when the system is in state $q_i$.


**References:**

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.

Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. 

Durbin, R., Eddy, S. R., Krogh, A., & Mitchison, G. (1998). Biological Sequence Analysis:


