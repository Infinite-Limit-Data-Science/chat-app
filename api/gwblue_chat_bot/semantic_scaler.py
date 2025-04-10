import math
from typing import List

class Sigmoid:
    @staticmethod
    def logistic(x: int, L: int, a: float, m: int) -> float:
        """
        logistic function
        $$
        \displaystyle \text{logistic}(x,L,a,m)
        \;=\;
        \frac{L}{1 + e^{-\,a\,(\,x - m\,)}}
        $$
        - x: input (context size)
        - L: Upper Bound (maximum value logistic approaches)
        - a: slope (e.g. 2 x 10^-5 for gradual expansion)
        - m: midpoint where logistic curve crosses L/2
        """
        return L / (1 + math.e**(-a * (x - m)))
    
def hamilton_method(demands: List[float], K_global: int):
    """
    largest remainder hamilton method
    \[ 
    k_i = \mathrm{round}\Bigl(\mathrm{Sigmoid.logistic}(x_i, L, a, m)\Bigr)
    \]
    $k_i =$ original value
    $k_i' =$ updated value
    \[
    k_{\text{sum}} = \sum_{i=1}^{N} k_i
    \]
    \[
    \text{If } \sum k_i \,\le\, K_{\max}, 
    \quad \text{then define } k_i' = k_i.
    \]
    \[
    \text{If } k_{\text{sum}} > K_{\max}, 
    \quad \text{then let } \alpha = \frac{K_{\max}}{\sum k_i}
    \quad 
    \]
    \[
    k_i' = \alpha \times k_i
    \]
    \[
    \text{base\_sum} 
    = \sum_{i=1}^{N} \lfloor k_i' \rfloor.
    \]
    - take each group's share
    - floor them to get a remainder
    - compute fractional remainders
    - sort descending by remainder
    - highest fractions first get additional seats
    """
    total_demands = sum(demands)
    if total_demands == 0:
        return [0]*len(demands)
    
    real_shares = [(d / total_demands) * K_global for d in demands]

    floor_vals = [math.floor(rs) for rs in real_shares]
    base_sum = sum(floor_vals)

    leftover = K_global - base_sum
    if leftover <= 0:
        return floor_vals

    remainders = [(i, real_shares[i] - floor_vals[i]) for i in range(len(demands))]
    remainders.sort(key=lambda x: x[1], reverse=True)

    final_counts = floor_vals[:]
    for i in range(leftover):
        idx = remainders[i][0]
        final_counts[idx] += 1

    return final_counts