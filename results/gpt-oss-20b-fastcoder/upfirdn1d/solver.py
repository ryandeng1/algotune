from typing import List, Tuple
import numpy as np
from scipy.signal import upfirdn

class Solver:
    """
    Optimize upfirdn computation by directly using the highly‑optimized
    scipy implementation.  The helper function is kept very small to avoid
    any overhead from additional Python logic.
    """
    
    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray, int, int]]) -> List[np.ndarray]:
        """
        Compute the upfirdn operation for each problem definition in the list.
        
        Parameters
        ----------
        problem : list of tuples
            Each tuple contains (h, x, up, down) where
             * h   : filter coefficients (1‑D array)
             * x   : input signal (1‑D array)
             * up  : upsampling factor (int)
             * down: downsampling factor (int)
        
        Returns
        -------
        list of np.ndarray
            The upfirdn results for each tuple in the input list.
        """
        # Using a list comprehension keeps the code concise and lets
        # Python internally optimise the loop.
        return [upfirdn(h, x, up=up, down=down) for h, x, up, down in problem]