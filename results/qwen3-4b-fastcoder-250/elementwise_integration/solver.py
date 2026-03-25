import numpy as np
from scipy.special import gamma

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        a = np.array(problem["a"])
        b = np.array(problem["b"])
        lower = np.array(problem["lower"])
        upper = np.array(problem["upper"])
        n = len(a)
        results = np.zeros(n)
        
        tol = 1e-12
        max_terms = 1000
        
        for i in range(n):
            a_i = a[i]
            b_i = b[i]
            lower_i = lower[i]
            upper_i = upper[i]
            
            total = 0.0
            k = 0
            while k < max_terms:
                numerator = upper_i**(k+1) - lower_i**(k+1)
                denom = np.factorial(k+1) * gamma(a_i * k + b_i)
                term = numerator / denom
                
                if abs(term) < tol:
                    break
                
                total += term
                k += 1
            results[i] = total
        
        return {"result": results.tolist()}
