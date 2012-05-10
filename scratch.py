# -*- coding: utf-8 -*-
import numpy as np
from scipy import weave

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