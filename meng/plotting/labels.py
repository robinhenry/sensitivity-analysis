
def magn_coeff(x_i, u_j, p_or_q, method):
    l = r'$\partial |V_{%d}|/\partial %s_{%d}$ (%s)' % (x_i, p_or_q, u_j, method)
    return l

def real_coeff(x_i, u_j, p_or_q, method):
    l = r'$\partial Re\{V_{%d}\} / \partial %s_{%d}$ (%s)' % (x_i, p_or_q, u_j, method)
    return l

def imag_coeff(x_i, u_j, p_or_q, method):
    l = r'$\partial Im\{V_{%d}\} / \partial %s_{%d}$ (%s)' % (x_i, p_or_q, u_j, method)
    return l
