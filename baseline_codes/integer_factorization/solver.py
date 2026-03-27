from typing import Any
import sympy


class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Reference solver using sympy.factorint.

        :param problem: A dictionary containing the composite number.
        :return: A dictionary with keys "p" and "q" containing the two prime factors, where p < q.
        :raises ValueError: If the factorization does not result in exactly two prime factors.
        """
        composite_val = problem["composite"]

        try:
            composite = sympy.Integer(composite_val)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"The composite value '{composite_val}' could not be converted to a SymPy Integer: {e}"
            )

        factors = [prime for prime, exp in sympy.factorint(composite).items() for _ in range(exp)]
        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, but got {len(factors)}.")

        p, q = sorted(map(int, factors))
        return {"p": p, "q": q}
