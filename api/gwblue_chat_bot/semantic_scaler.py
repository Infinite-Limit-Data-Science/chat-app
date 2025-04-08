import math

class Sigmoid:
    @staticmethod
    def logistic(x: int, L: int, a: float, m: int) -> float:
        """
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

