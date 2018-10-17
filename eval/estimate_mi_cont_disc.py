import numpy as np

# a is continuous, b is discrete
def calculate_mi(a,b,bins=20):
    a_vals, a_bins = np.histogram(a, bins=bins)
    b_vals = np.unique(b)
    mi=0.
    n = len(a)
    epsilon=1e-4
    for b_val in b_vals:
        b_match = (b==b_val)
        for a_l, a_r in zip(a_bins[:-1],a_bins[1:]):
            if a_r != a_bins[-1]:
                a_match = (a >= a_l)&(a < a_r)
            else:
                a_match = (a >= a_l)&(a <= a_r)
            p_ab = float( (a_match&b_match).sum() ) / n
            p_a = float(a_match.sum())/n
            p_b = float(b_match.sum())/n
            if p_a != 0. and p_b != 0. and p_ab > epsilon:
                #print a_l, a_r, b_val, p_ab, p_a, p_b, p_ab/(p_a*p_b)
                mi += p_ab * np.log2( (p_ab) /(p_a*p_b))
    return mi
