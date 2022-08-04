import numpy as np

def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))

def annealing(bounds,
              cost_function,
              random_neighbour,
              temperature_fn=temperature,
              maxsteps=1000,
              debug=True):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state = random_start(bounds)
    cost = cost_function(state)
    states, costs = [state], [cost]
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature_fn(fraction)
        new_state = random_neighbour(state, bounds, fraction)
        new_cost = cost_function(new_state)
        if debug and step % 1 == 0:
            print(f"Step #{step}/{maxsteps} : T = {T}, state = {state}, cost = {cost[0]:8.4}, new_state = {new_state}, new_cost = {new_cost[0]:8.4} ...")
        if acceptance_probability(cost, new_cost, T) > np.random.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
            # print("  ==> Accept it!")
        # else:
        #    print("  ==> Reject it...")
    return state, cost_function(state), states, costs


def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p


def random_start(interval):
    """ Random point in the interval."""
    a, b = interval
    return a + int((b - a) * np.random.random_sample())

