"""
The script used in this video presentation:
https://youtu.be/E7V36JvHbsA
Relevant Wikipedia pages:
https://en.wikipedia.org/wiki/Exponential_function
https://en.wikipedia.org/wiki/Matrix_exponential
https://en.wikipedia.org/wiki/Logarithm_of_a_matrix
https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula
$\mathrm{exp}(sX)* \mathrm{exp}(tY)=\mathrm{exp}\left(sX+tY+st\frac{1}{2}[X,Y]+s^2t\frac{1}{12}[X,[X,Y]]-st^2\frac{1}{12}[Y,[X,Y]]+\cdots\right)$
Let ${\mathcal N}_n$ be the set of lists, of length n, of pairs of naturals, but excluding the origin $(0,0)$.
For one usage, consider e.g. the factorial coefficients in (for $r>0$)
$(a+b)^r = \sum_{k=0}^r \frac{r!}{k!(r-k)!}a^kb^{r-k} = r! \sum_{(k,j)\in{\mathcal N}_1, k+j=r} \frac{1}{k!\,j!}a^kb^j$
What we have is
$-\sum_{n=1}^r\frac{(-1)^n}{n}\sum_{(k, h)_n\in{\mathcal N}_n, \sum_{i=1}^n(k_i+j_i)=r}\frac{1}{\prod_{i=1}^n k_i!\cdot h_i!}\left(\prod_{i=1}^n u^{k_i}v^{h_i}-\frac{1}{r}[\cdots, [\dots, v]\cdots]_{k,h}\right)=0$
Note: For a Lie bracket given in an $m$-dimensional vector space, there's only $l=\tfrac{1}{2}m(m-1)$ non-necessarily-zero lookup-relations $[X_i, X_j]=\sum_{k=1,2,3}c_{i,j,k}X_k$.
E.g. for $3$-dimensional space with basis $\{X_1,X_2,X_3\}$, there's $[X_1, X_2], [X_1, X_3]$ and $[X_2, X_2]$, i.e. $l=3$.
For general $d\times d$-matrices, we have $m=d^2$, so $l={\mathcal O}(d^4)$.
"""

import numpy as np
from scipy.linalg import expm, logm
from termcolor import colored


TOLERANCE = 1e-13


def log(color, message_string):
    print(colored(message_string, color))


# $[X, Y] := X\cdot Y - Y\cdot X$
def comm(x, y):
    x = np.array(x)
    y = np.array(y)
    return x.dot(y) - y.dot(x)


# $-(-1)^n \cdot n\cdot \prod_{i=1}^n k_i!\cdot h_i!$
def denom(pl):
    #print('denom_pl\n', pl)
    n = len(pl)
    #print('denom_n\n', n)
    res = (-1)**(n-1) * n
    for p in pl:
        for e in p:
            #print('e in p', e)
            res *= np.math.factorial(e)
    return res


# $ \prod_{i=1}^n u^{k_i}v^{h_i} $
def pl_prod(pl, u, v):
    res = np.identity(u.shape[0])
    '''print('pl_prod.pl\n', pl)
    print('pl_prod.u\n', u)
    print('pl_prod.v\n', v)
    print('pl_res\n', res)'''
    for p in pl:
        ews = [(p[0], u), (p[1], v)]
        #print('ews\n', ews)
        for e, w in ews:
            #print('e\n', e)
            #print('w\n', w)
            for _ in range(e):
                res = res.dot(w)
                #print('res after op, \n', res)
    return res


# $ [\cdots, [\dots, v]\cdots]_{k,h} $
def pl_comm(pl, u, v):
    res = None
    for pair in pl[::-1]:
        ews = [(pair[0], u), (pair[1], v)]
        for e, w in ews[::-1]:
            if e: # TODO: check if [w^0, foo]=0 is considered or skipped
                if res is None:
                    if e == 1:
                        res = w
                    else:
                        return np.zeros(w.shape) # [w, w]
                else:
                    if e:
                        for _ in range(e):
                            res = comm(w, res)
    return res


def abs_elem_sum(x):
    res = 0
    for row in x:
        for elem in row:
            res += abs(elem)
    return res


def Z(a, b, joint_sum, left_sum=1):
    ps = [ (i, j) for i in range(joint_sum + 1) for j in range(joint_sum + 1) if i+j and (i+j <= joint_sum) ]
    print('JOINT SUM: ', joint_sum)
    print('ps\n', ps)

    pls = [[]]
    for ii in range(joint_sum):
        pls += [pl + [p] for pl in pls for p in ps]
    #print('PLS\n', pls)

    res_prod, res_comm, res_prod_, res_comm_ = np.zeros(a.shape), np.zeros(a.shape), np.zeros(a.shape), np.zeros(a.shape)

    used_pls = []
    used_pls_ = []

    for pl in pls:
        #print('pl in pls\n', pl)
        if sum([p[0] + p[1] for p in pl]) == joint_sum and pl not in used_pls:
            used_pls.append(pl)

            t_prod = pl_prod(pl, a, b) / denom(pl)
            #print('denom_pl\n', denom(pl))
            #print('t_prod\n', t_prod)

            # $ \sum_{i=1}^n (|k_i| + |h_i|) $
            t_comm = pl_comm(pl, a, b) / denom(pl) / joint_sum
            #print('t_comm\n', t_comm)
            res_prod += t_prod
            res_comm += t_comm

            if sum([p[0] for p in pl]) == left_sum:
                used_pls_.append(pl)
                res_prod_ += t_prod
                res_comm_ += t_comm

            #print("\t- comm denom = {}".format(denom(pl) * joint_sum))


    '''log("yellow", 40 * "-" + "<log Z with joint_sum={}, left_sum={}>".format(joint_sum, left_sum))
    log("yellow", "len(ps):\t\t{}".format(len(ps)))
    log("yellow", "len(pls):\t\t{}".format(len(pls)))
    log("yellow", "len(used_pls):\t{}".format(len(used_pls)))
    log("yellow", "len(used_pls_):\t{}".format(len(used_pls_)))
    log("yellow", "ps:\t\t{}".format(ps))
    #log("yellow", "used_pls:\t\t{}".format(used_pls))
    log("yellow", "used_pls_:\t\t{}".format(used_pls_))
    log("yellow", "res_prod =\n{}".format(res_prod))
    log("yellow", "res_prod_ =\n{}".format(res_prod_))
    log("yellow", 40 * "-" + "</log>")'''
    assert abs_elem_sum(res_prod - res_comm) < TOLERANCE, abs_elem_sum(res_prod - res_comm)
    assert abs_elem_sum(res_prod_ - res_comm_) < TOLERANCE, abs_elem_sum(res_prod_ - res_comm_)

    return res_prod


def Z_sum(a, b, n):
    #log("cyan", 50 * "=" + "<log> Z sum with n = {}".format(n))

    Z1 = Z(a, b, 1)
    #print('Z1: ', Z1)
    #log("cyan", "a+b =\n{}".format(a+b))

    Z2 = Z(a, b, 2)
    print('Z2: ', Z2)
    #log("cyan", "comm(a,b) / 2 =\n{}".format(comm(a,b)/2))

    res = Z1 + Z2
    for idx in range(3, n): # up to 6 is okay w.r.t. computation time
        #print('idx_x: ', idx)
        '''print(80 * '-')
        print('idx: ')
        print(80 * '-')'''
        res += Z(a, b, idx, 1)

    #log("cyan", 50 * "=" + "</log> Z sum with n = {}".format(n))

    return res


if __name__=="__main__":
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)
    a = np.identity(3)
    b = np.identity(2)
    b = np.array([[5,2,3], [1,4,3],[3,3,3]])
    print(a)
    print(b)
    #assert abs_elem_sum(logm(expm(a))-a) < TOLERANCE, abs_elem_sum(logm(expm(a))-a)
    #assert abs_elem_sum(logm(expm(b))-b) < TOLERANCE, abs_elem_sum(logm(expm(b))-a)

    #print(80 * "#" + "\n" +  80 * "#")

    #Z(a, b, 3, 2) # up to 5 is okay w.r.t. computation time
    #exit()

    Znumpy = logm(expm(a).dot(expm(b)))
    print('Znumpy', Znumpy)
    Z5 = Z_sum(a, b, 3)
    print('Z5', Z5)
    diff = Znumpy - Z5
    print('diff', diff)

    print("a =\n{}".format(a))
    print("b =\n{}\n".format(b))
    log("green", "Znumpy =\n{}".format(Znumpy))
    log("green", "Z5 =\n{}\n".format(Z5))
    log("green", "Znumpy - Z5 =\n{}\nabs_elem_sum(Znumpy-Z5) = {}".format(Znumpy-Z5, abs_elem_sum(Znumpy-Z5)))
