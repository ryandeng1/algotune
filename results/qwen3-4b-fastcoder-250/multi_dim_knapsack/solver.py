class Solver:
    def solve(self, problem):
        value, demand, supply = problem
        n = len(value)
        k = len(supply)
        
        dp = {(0,) * k: 0}
        parent = {(0,) * k: (None, None)}
        
        for i in range(n):
            new_dp = dp.copy()
            new_parent = parent.copy()
            for state, val in dp.items():
                new_state = tuple(state[j] + demand[i][j] for j in range(k))
                if all(new_state[j] <= supply[j] for j in range(k)):
                    new_val = val + value[i]
                    if new_state not in new_dp or new_val > new_dp[new_state]:
                        new_dp[new_state] = new_val
                        new_parent[new_state] = (state, i)
            dp = new_dp
            parent = new_parent
        
        best_state = max(dp.items(), key=lambda x: x[1])[0]
        selected = []
        state = best_state
        while parent[state][0] is not None:
            prev_state, item_idx = parent[state]
            selected.append(item_idx)
            state = prev_state
        return selected[::-1]
