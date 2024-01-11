"""
  Our script is obtained by adding the NN algorithm to Esser-Zweydinger’s script。
  ”bjmm_depth_2_qc_complexity“ is the estimator of Esser-Zweydinger’s algorithm.
  Our estimation program is defined as "new_bjmm_depth_2_qc_complexity".
  Esser-Zweydinger’s algorithm is can be obtained in the reference [26], and their script is available at
  https://github.com/FloydZ/Improving-ISD-in-Theory-and-Practice.
       [26] Esser, A., Zweydinger, F.: New time-memory trade-offs for subset sum improving ISD in theory and practice.
            In: Hazay, C., Stam, M. (eds.) EUROCRYPT 2023, vol. 14008, pp. 360–390.
"""
from math import comb as binom
from math import log2, log, ceil, inf, floor
from scipy.optimize import fsolve

def H(x):
    h = -x * log2(x) - (1-x) * log2(1-x)
    return h

def Hi(v):
    if v == 1:
        return 0.5
    if v < 0.00001:
        return 0

    return fsolve(lambda x: v -(-x*log2(x)-(1-x)*log2(1-x)), 0.0000001)[0]

###############################################################################
#######################Adapted estimator functions#############################
###############################################################################
def _list_merge_async_complexity(L1, L2, l, hmap):
    """
    Complexity estimate of merging two lists exact

    INPUT:

    - ``L`` -- size of lists to be merged
    - ``l`` -- amount of bits used for matching
    - ``hmap`` -- indicates if hashmap is being used (Default 0: no hashmap)
    """

    if L1 == 1 and L2== 1:
        return 1
    if L1 == 1:
        return L2
    if L2 == 1:
        return L1
    if not hmap:
        return 0 #to be implemented
    else:
        return L1+L2 + L1*L2 // 2 ** l


def _list_merge_complexity(L, l, hmap):
    """
    Complexity estimate of merging two lists exact
    INPUT:
    - ``L`` -- size of lists to be merged
    - ``l`` -- amount of bits used for matching
    - ``hmap`` -- indicates if hashmap is being used (Default 0: no hashmap)
    """

    if L == 1:
        return 1
    if not hmap:
        return max(1, 2 * int(log2(L)) * L + L ** 2 // 2 ** l)
    else:
        return 2 * L + L ** 2 // 2 ** l


def _gaussian_elimination_complexity(n, k, r):
    """
    Complexity estimate of Gaussian elimination routine
    INPUT:
    - ``n`` -- Row additons are perfomed on ``n`` coordinates
    - ``k`` -- Matrix consists of ``n-k`` rows
    - ``r`` -- Blocksize of method of the four russian for inversion, default
                is zero
    [Bar07]_ Bard, G.V.: Algorithms for solving linear and polynomial systems
    of equations over finite fields
    with applications to cryptanalysis. Ph.D. thesis (2007)
    [BLP08] Bernstein, D.J., Lange, T., Peters, C.: Attacking and defending the
    mceliece cryptosystem.
    In: International Workshop on Post-Quantum Cryptography. pp. 31–46.
    Springer (2008)
    """

    if r != 0:
        return (r ** 2 + 2 ** r + (n - k - r)) * int(((n + r - 1) / r))

    return (n - k) ** 2


def _optimize_m4ri(n, k, mem=inf):
    """
    Find optimal blocksize for Gaussian elimination via M4RI
    INPUT:
    - ``n`` -- Row additons are perfomed on ``n`` coordinates
    - ``k`` -- Matrix consists of ``n-k`` rows
    """

    (r, v) = (0, inf)
    for i in range(n - k):
        tmp = log2(_gaussian_elimination_complexity(n, k, i))
        if v > tmp and r < mem:
            r = i
            v = tmp
    return r


def _mem_matrix(n, k, r):
    """
    Memory usage of parity check matrix in vector space elements
    INPUT:
    - ``n`` -- length of the code
    - ``k`` -- dimension of the code
    - ``r`` -- block size of M4RI procedure
    """
    return n - k + 2 ** r



# def bjmm_depth_2_qc_complexity(n: int, k: int, w: int, mem=inf, hmap=1, mmt=0, qc=0, base_p=-1, l_val=0, l1_val=0, memory_access=0, enable_tmto=1):
def bjmm_depth_2_qc_complexity(n: int, k: int, w: int, mem=inf, hmap=1, mmt=0, qc=0, base_p=-1, l_val=0, l1_val=0, memory_access=0, enable_tmto=1):
    """
        Complexity estimate of BJMM algorithm in depth 2
        [MMT11] May, A., Meurer, A., Thomae, E.: Decoding random linear codes in  2^(0.054n). In: International Conference
        on the Theory and Application of Cryptology and Information Security. pp. 107–124. Springer (2011)
        [BJMM12] Becker, A., Joux, A., May, A., Meurer, A.: Decoding random binary linear codes in 2^(n/20): How 1+ 1= 0
        improves information set decoding. In: Annual international conference on the theory and applications of
        cryptographic techniques. pp. 520–536. Springer (2012)
        expected weight distribution::
            +--------------------------+-------------------+-------------------+
            | <-----+ n - k - l +----->|<--+ (k + l)/2 +-->|<--+ (k + l)/2 +-->|
            |           w - 2p         |        p          |        p          |
            +--------------------------+-------------------+-------------------+

        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2), default unlimited
        - ``hmap`` -- indicates if hashmap is being used (default: true)
        - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
        - ``mmt`` -- restrict optimization to use of MMT algorithm (precisely enforce p1=p/2)
        - ``qc`` -- optimize in the quasi cyclic setting
        - ``base_p`` -- hard code the base p enumerated in the baselists.
                        if this value is set to -1 the code will optimize over
                        different values
        - ``l_val`` -- hardcode `l`. If set to 0 the code will optimize over
                        different values.
        - ``l1_val`` -- same as `l` only for `l1`
        - ``memory_access`` -- specifies the memory access cost model
                                (default: 0, choices:
                                 0 - constant,
                                 1 - logarithmic,
                                 2 - square-root,
                                 3 - cube-root
                                 or deploy custom function which takes as input
                                 the logarithm of the total memory usage)
        - ``enable_tmto`` -- enable the new time memory tradeoff proposed in
                            this work
    """
    ### qc=0,McEliece; qc=1,HQC & BIKE


    n = int(n)
    k = int(k)
    w = int(w)

    solutions = max(0, log2(binom(n, w)) - (n - k))
    time = inf
    memory = 0
    r = _optimize_m4ri(n, k, mem)

    i_val = [25, 450, 25]
    i_val_inc = [10, 10, 10]
    params = [-1 for _ in range(7)]
    lists = []

    while True:
        stop = True
        mod2 = (params[0] - i_val_inc[0]//2) % 2
        for p in range(max(params[0] - i_val_inc[0]//2 - mod2, 2*qc), min(w // 2, i_val[0]), 2):
            for l in range(max(params[1] - i_val_inc[1] // 2, 0), min(n - k - (w - 2 * p), min(i_val[1], n - k))):
                for p1 in range(max(params[2] - i_val_inc[2] // 2, (p + 1) // 2, qc), min(w, i_val[2])):
                    if mmt and p1 != p // 2:
                        continue

                    if base_p != -1 and p1 != base_p:
                        continue

                    if l_val != 0 and l != l_val:
                        continue

                    k1 = (k + l) // 2
                    L1 = binom(k1, p1)
                    if qc:
                        L1b = binom(k1, p1-1)*k

                    if log2(L1) > time:
                        continue

                    if k1 - p < p1 - p / 2:
                        continue

                    if not (qc):
                        reps = (binom(p, p//2) * binom(k1 - p, p1 - p//2)) ** 2
                    else:
                        reps = binom(p, p//2) * binom(k1 - p, p1 - p//2) * binom(k1 - p+1, p1 - p // 2)
                        if p-1 > p//2:
                            reps *= (binom(p-1, p // 2))

                    if enable_tmto == 1:
                        start = int(log2(L1))-5
                        end = start + 10
                    else:
                        start = int(ceil(log2(reps)))
                        end = start + 1

                    for l1 in range(start, end):
                        if l1 > l:
                            continue

                        L12 = max(1, L1 ** 2 // 2 ** l1)

                        qc_advantage = 0
                        if qc:
                            L12b = max(1, L1*L1b//2**l1)
                            qc_advantage = log2(k)

                        tmp_mem = log2((2 * L1 + L12) + _mem_matrix(n, k, r)) if not (
                            qc) else log2(L1+L1b + min(L12, L12b) + _mem_matrix(n, k, r))
                        if tmp_mem > mem:
                            continue

                        Tp = max(log2(binom(n, w))
                                 - log2(binom(n - k - l, w - 2 * p + qc))
                                 - log2(binom(k1, p))
                                 - log2(binom(k1, p - qc))
                                 - qc_advantage - solutions, 0)

                        Tg = _gaussian_elimination_complexity(n, k, r)
                        if not (qc):
                            T_tree = 2 * _list_merge_complexity(L1, l1, hmap) + _list_merge_complexity(L12,
                                                                                                       l - l1,
                                                                                                       hmap)
                        else:
                            T_tree = _list_merge_async_complexity(L1, L1b, l1, hmap) + _list_merge_complexity(L1, l1, hmap) + _list_merge_async_complexity(L12, L12b,
                                                                                                                                                           l-l1, hmap)

                        T_rep = int(ceil(2 ** (l1 - log2(reps))))

                        tmp = Tp + log2(Tg + T_rep * T_tree)
                        # print(tmp, Tp, T_rep, T_tree)

                        if memory_access == 1:
                            tmp += log2(tmp_mem)
                        elif memory_access == 2:
                            tmp += tmp_mem/3
                        elif callable(memory_access):
                            tmp += memory_access(tmp_mem)

                        if tmp < time or (tmp == time and tmp_mem < memory):
                            time = tmp
                            memory = tmp_mem
                            params = [p, l, p1, T_tree, Tp, l1, log2(reps)]
                            tree_detail = [log2(Tg), log2(
                                2 * _list_merge_complexity(L1, l1, hmap)), log2(_list_merge_complexity(L12, l - l1, hmap))]
                            lists = [log2(L1), log2(L12), 2*log2(L12)-(l-l1)]

        for i in range(len(i_val)):
            if params[i] == i_val[i] - 1:
                stop = False
                i_val[i] += i_val_inc[i]

        if stop == True:
            break
    par = {"l": params[1], "p": params[0], "p1": params[2],
           "l1": params[5], "reps": params[6], "depth": 2, "v": floor(params[5]-params[6])}
    # res = {"time": time, "memory": memory, "parameters": par,
    #        "perms": params[4], "lists": lists}
    res = [time, memory, par, lists]
    return res


def new_bjmm_depth_2_qc_complexity(n: int, k: int, w: int, mem=inf, hmap=1, mmt=0, qc=0, base_p=-1, l_val=0, l1_val=0, memory_access=0, enable_tmto=1):
    """
        Complexity estimate of BJMM algorithm in depth 2
        [MMT11] May, A., Meurer, A., Thomae, E.: Decoding random linear codes in  2^(0.054n). In: International Conference
        on the Theory and Application of Cryptology and Information Security. pp. 107–124. Springer (2011)
        [BJMM12] Becker, A., Joux, A., May, A., Meurer, A.: Decoding random binary linear codes in 2^(n/20): How 1+ 1= 0
        improves information set decoding. In: Annual international conference on the theory and applications of
        cryptographic techniques. pp. 520–536. Springer (2012)
        expected weight distribution::
            +--------------------------+-------------------+-------------------+
            | <-----+ n - k - l +----->|<--+ (k + l)/2 +-->|<--+ (k + l)/2 +-->|
            |           w - 2p         |        p          |        p          |
            +--------------------------+-------------------+-------------------+

        INPUT:
        - ``n`` -- length of the code
        - ``k`` -- dimension of the code
        - ``w`` -- Hamming weight of error vector
        - ``mem`` -- upper bound on the available memory (as log2), default unlimited
        - ``hmap`` -- indicates if hashmap is being used (default: true)
        - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
        - ``mmt`` -- restrict optimization to use of MMT algorithm (precisely enforce p1=p/2)
        - ``qc`` -- optimize in the quasi cyclic setting
        - ``base_p`` -- hard code the base p enumerated in the baselists.
                        if this value is set to -1 the code will optimize over
                        different values
        - ``l_val`` -- hardcode `l`. If set to 0 the code will optimize over
                        different values.
        - ``l1_val`` -- same as `l` only for `l1`
        - ``memory_access`` -- specifies the memory access cost model
                                (default: 0, choices:
                                 0 - constant,
                                 1 - logarithmic,
                                 2 - square-root,
                                 3 - cube-root
                                 or deploy custom function which takes as input
                                 the logarithm of the total memory usage)
        - ``enable_tmto`` -- enable the new time memory tradeoff proposed in
                            this work
    """

    n = int(n)
    k = int(k)
    w = int(w)

    solutions = max(0, log2(binom(n, w)) - (n - k))
    time = inf
    memory = 0
    r = _optimize_m4ri(n, k, mem)

    i_val = [25, 450, 25]
    i_val_inc = [10, 10, 10]
    params = [-1 for _ in range(7)]
    lists = []

    while True:
        stop = True
        mod2 = (params[0] - i_val_inc[0]//2) % 2
        for p in range(max(params[0] - i_val_inc[0]//2 - mod2, 2*qc), min(w // 2, i_val[0]), 2):
            for l in range(max(params[1] - i_val_inc[1] // 2, 0), min(n - k - (w - 2 * p), min(i_val[1], n - k))):
                for p1 in range(max(params[2] - i_val_inc[2] // 2, (p + 1) // 2, qc), min(w, i_val[2])):
                    if mmt and p1 != p // 2:
                        continue

                    if base_p != -1 and p1 != base_p:
                        continue

                    if l_val != 0 and l != l_val:
                        continue

                    k1 = (k + l) // 2
                    L1 = binom(k1, p1)
                    if qc:
                        L1b = binom(k1, p1-1)*k

                    if log2(L1) > time:
                        continue

                    if k1 - p < p1 - p / 2:
                        continue

                    if not (qc):
                        reps = (binom(p, p//2) * binom(k1 - p, p1 - p//2)) ** 2
                    else:
                        reps = binom(p, p//2) * binom(k1 - p, p1 - p//2) * binom(k1 - p+1, p1 - p // 2)
                        if p-1 > p//2:
                            reps *= (binom(p-1, p // 2))

                    if enable_tmto == 1:
                        start = int(log2(L1))-5
                        end = start + 10
                    else:
                        start = int(ceil(log2(reps)))
                        end = start + 1

                    for l1 in range(start, end):
                        if l1 == l:
                            L12 = max(1, L1 ** 2 // 2 ** l1)

                            qc_advantage = 0
                            if qc:
                                L12b = max(1, L1*L1b//2**l1)
                                qc_advantage = log2(k)

                            tmp_mem = log2((2 * L1 + L12) + _mem_matrix(n, k, r)) if not (
                                qc) else log2(L1+L1b + min(L12, L12b) + _mem_matrix(n, k, r))
                            if tmp_mem > mem:
                                continue

                            Tp = max(log2(binom(n, w))
                                     - log2(binom(n - k - l, w - 2 * p + qc))
                                     - log2(binom(k1, p))
                                     - log2(binom(k1, p - qc))
                                     - qc_advantage - solutions, 0)

                            Tg = _gaussian_elimination_complexity(n, k, r)

                            lamda = log2(L12) / (n - k - l)
                            delta = Hi(1 - lamda)
                            gamma = (w - p - p) / (n - k - l)
                            gamma1 = 2 * delta * (1 - delta)
                            gamma2 = gamma / 2

                            if delta - gamma2 > 0 and gamma <= gamma1:
                                t_NN = (1 - gamma) * (1 - H((delta - gamma2) / (1 - gamma))) * (n - k - l)
                            if gamma > gamma1:
                                t_NN = 2 * lamda + H(gamma) - 1

                            if not (qc):
                                T_tree = 2 * _list_merge_complexity(L1, l1, hmap) + t_NN
                            else:
                                T_tree = _list_merge_async_complexity(L1, L1b, l1, hmap) + _list_merge_complexity(L1, l1, hmap) + t_NN

                            T_rep = int(ceil(2 ** (l1 - log2(reps))))

                            tmp = Tp + log2(Tg + T_rep * T_tree)
                            # print(tmp, Tp, T_rep, T_tree)

                            if memory_access == 1:
                                tmp += log2(tmp_mem)
                            elif memory_access == 2:
                                tmp += tmp_mem/3
                            elif callable(memory_access):
                                tmp += memory_access(tmp_mem)

                            if tmp < time or (tmp == time and tmp_mem < memory):
                                time = tmp
                                memory = tmp_mem
                                params = [p, l, p1, T_tree, Tp, l1, log2(reps)]
                                tree_detail = [log2(Tg), log2(
                                    2 * _list_merge_complexity(L1, l1, hmap)), log2(_list_merge_complexity(L12, l - l1, hmap))]
                                lists = [log2(L1), log2(L12)]

        for i in range(len(i_val)):
            if params[i] == i_val[i] - 1:
                stop = False
                i_val[i] += i_val_inc[i]

        if stop == True:
            break
    par = {"l": params[1], "p": params[0], "p1": params[2],
           "l1": params[5], "reps": params[6], "depth": 2}
    # res = {"time": time, "memory": memory, "parameters": par,
    #        "perms": params[4], "lists": lists}
    res = [time, memory, par, lists]
    return res

def ISD_estimator(name, n, k, t, qc, max_mem):
    mem = max_mem - ceil(log2(n))

    paremeters = [n, k, t]

    ##### constant ####
    T1 = bjmm_depth_2_qc_complexity(n, k, t, mem, 1, 0, qc, -1, 0, 0, 0, 1)
    T2 = new_bjmm_depth_2_qc_complexity(n, k, t, mem, 1, 0, qc, -1, 0, 0, 0, 1)

    #### logarithmic ####
    T11 = bjmm_depth_2_qc_complexity(n, k, t, mem, 1, 0, qc, -1, 0, 0, 1, 1)
    T21 = new_bjmm_depth_2_qc_complexity(n, k, t, mem, 1, 0, qc, -1, 0, 0, 1, 1)

    #### cube-root ####
    T12 = bjmm_depth_2_qc_complexity(n, k, t, mem, 1, 0, qc, -1, 0, 0, 2, 1)
    T22 = new_bjmm_depth_2_qc_complexity(n, k, t, mem, 1, 0, qc, -1, 0, 0, 2, 1)

    print("    ")
    print("=============================================")
    print(f"{name :^41}")
    print("---------------------------------------------")
    print(f"{'[n, k, t] =' :>20}", paremeters)
    print(f"{'Memory <=' :>20}", max_mem, "bits")
    print("---------------------------------------------")
    print(f"{'Technique' :<10} {'constant' :^12} {'logarithmic' :<12} {'cube-root' :<12}")
    print(f"{'[28]' :<10} {'%.2f' % (T1[0] + log2(n)) :^12} {'%.2f' % (T11[0] + log2(n)) :^12}"
          f"{'%.2f' % (T12[0] + log2(n)) :^12}")
    print(f"{'Ours' :<10} {'%.2f' % (T2[0] + log2(n)) :^12} {'%.2f' % (T21[0] + log2(n)) :^12}"
          f"{'%.2f' % (T22[0] + log2(n)) :^12}")
    print("=============================================")


# Sheme = {"name": "McEliece: 128-bit", "n": 3488, "k": 2720, "t": 64, "qc": 0}
# Sheme = {"name": "McEliece: 192-bit", "n": 4608, "k": 3360, "t": 96, "qc": 0}
# Sheme = {"name": "McEliece: 256-bit", "n": 6688, "k": 5024, "t": 128, "qc": 0}
# Sheme = {"name": "McEliece: 256-bit", "n": 6960, "k": 5413, "t": 119, "qc": 0}
# Sheme = {"name": "McEliece: 256-bit", "n": 8192, "k": 6528, "t": 128, "qc": 0}
#
# Sheme = {"name": "BIKE_message: 128-bit", "n": 24646, "k": 12323, "t": 134, "qc": 1}
# Sheme = {"name": "BIKE_message: 192-bit", "n": 49318, "k": 24659, "t": 199, "qc": 1}
Sheme = {"name": "BIKE_message: 256-bit", "n": 81946, "k": 40973, "t": 264, "qc": 1}

# Sheme = {"name": "BIKE_key: 128-bit",     "n": 24646, "k": 12323, "t": 142, "qc": 1}
# Sheme = {"name": "BIKE_key: 192-bit",     "n": 49318, "k": 24659, "t": 206, "qc": 1}
# Sheme = {"name": "BIKE_key: 256-bit",     "n": 81946, "k": 40973, "t": 274, "qc": 1}
#
# Sheme = {"name": "HQC: 128-bit", "n": 35338,  "k": 17669, "t": 132, "qc": 1}
# Sheme = {"name": "HQC: 192-bit", "n": 71702,  "k": 35851, "t": 200, "qc": 1}
# Sheme = {"name": "HQC: 256-bit", "n": 115274, "k": 57637, "t": 262, "qc": 1}

ISD_estimator(Sheme["name"], Sheme["n"], Sheme["k"], Sheme["t"], Sheme["qc"], 60)
