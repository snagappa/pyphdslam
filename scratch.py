# -*- coding: utf-8 -*-
import numpy as np
from scipy import weave
import phdslam

import gmphdfilter
import phdmisctools

def test_slam_feature(num_x=5, dim_x=2):
    x, mu, P = phdmisctools.test_data(num_x, dim_x)
    states = [[x[i], P[i]] for i in range(num_x)]
    weights = np.random.rand(num_x)
    
    default_args = gmphdfilter.get_default_gmphd_obj_args()
    slam_feature = phdslam.GMPHD_SLAM_FEATURE(*default_args)
    # Disable birth
    slam_feature.birth_fn.handle = None
    slam_feature.init(states, weights)
    slam_feature.phdIterate(mu)
    return slam_feature






def _compute_residuals_(x, mu):
    num_x = len(x)
    num_mu = len(mu)
    state_dimensions = len(mu[0])
    if num_x == 1:
        residuals = np.array([np.empty(state_dimensions, dtype=np.float64) for i in range(num_mu)])
    else:
        residuals = np.array([np.empty(state_dimensions, dtype=np.float64) for i in range(num_x)])
    num_residuals = [0];
    python_vars = ['x', 'mu', 'residuals', 'num_residuals']
    
    code2 = """
    int num_x = Nx[0];
    int num_mu = Nmu[0];
    
    int state_dimensions = Nmu[1];
    int icnt, jcnt;
    
    
    // Compute the residuals
    if (num_x == 1) {
        for (icnt=0; icnt<num_mu; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals(icnt,jcnt) = x(0, jcnt)-mu(icnt,jcnt);
            }
        }
        num_residuals[0] = num_mu;
    }
    else if (num_mu == 1) {
        for (icnt=0; icnt<num_x; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals(icnt,jcnt) = x(icnt,jcnt)-mu(0,jcnt);
            }
        }
        num_residuals[0] = num_x;
    }
    else {
        for (icnt=0; icnt<num_x; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals(icnt,jcnt) = x(icnt,jcnt)-mu(icnt,jcnt);
            }
        }
        num_residuals[0] = num_x;
    }
    
    """
    weave.inline(code2, python_vars, type_converters=weave.converters.blitz)
    return residuals, num_residuals[0]









code = """
double t[4];
std::string tab = " ";
t[0] = seq[n][0];
t[1] = seq[n][1];
t[2] = seq[n][2];
t[3] = seq[n][3];
std::cout << t[0] << tab << t[1] << tab << t[2] << tab << t[3] << std::endl;
//std::cout << "List length = " << seq.length() << std::endl;
//std::cout << "State length = " << seq[0].length() << std::endl;
seq[n][3] = t[0]+t[1]+t[2]+t[3];
std::cout << "seq[n][3]=" << double(seq[n][3]) << std::endl;
"""

"""seq = [np.random.rand(4) for count in range(3)]
n = 0
weave.inline(code, ['seq', 'n'], verbose=1, compiler='gcc')
print seq[n]"""


imul_code = """
for (int i=0; i<vec_len;i++) {
    vec[i] *= mul_fac;
}
"""

def imul(vec, mul_fac):
    vec_len = len(vec)
    weave.inline(imul_code, ['vec', 'mul_fac', 'vec_len'])
    
    
class gsl_test:
    code = """
gsl_vector_int_view gsl_v = gsl_vector_int_view_array( &seq[0][0], seq[0].length() );
//gsl_linalg_cholesky_decomp (gsl_matrix * A);

for (int c = 0; c < seq[0].length(); c++) {
    std::cout << seq[0][c] << " ";
}
std::cout << std::endl;
for (int c = 0; c < seq[0].length(); c++) {
    std::cout << gsl_v.vector.data[c] << " ";
}
std::cout << std::endl;
"""
    
    headers = ["<gsl/gsl_linalg.h>", "<gsl/gsl_vector.h>", "<gsl/gsl_matrix.h>"]
    libraries = ["gsl"]