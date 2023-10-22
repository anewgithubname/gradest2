/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "../cpp/juzhen.hpp" 
#include "../ml/gradest_fdiv.cuh"

/* Input Arguments */

#define XP_IN prhs[0] 
#define XQ_IN prhs[1]
#define XT_IN prhs[2]
#define SIGMA_CHOSEN prhs[3]
#define LAMBDA_CHOSEN prhs[4]

/* Output Arguments */
#define GRAD_OUT plhs[0]
#define SIGMA_OUT plhs[1]

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{   
    std::cout.precision(5);
#ifdef DEBUG
    std::cout << "num of input: " << nrhs << std::endl;
    std::cout << "num of output: " << nlhs << std::endl;
#endif

    size_t d, np, nq, nt;
    np = mxGetM(XP_IN);
    d = mxGetN(XP_IN);    
    nq = mxGetM(XQ_IN);
    nt = mxGetM(XT_IN);

    float sigma_chosen = mxGetScalar(SIGMA_CHOSEN);
    float lambda_chosen = mxGetScalar(LAMBDA_CHOSEN);
    
#ifdef DEBUG
    std::cout << "d: " << d << std::endl;
    std::cout << "np: " << np << std::endl;
    std::cout << "nq: " << nq << " nt: " << nt << std::endl;
    std::cout << "sigma: " << sigma_chosen << " lambda: " << lambda_chosen << std::endl;
#endif

    {
        CuBLASErrorCheck(cublasCreate(&Matrix<CUDAfloat>::global_handle));
        Memory<int> mdi;
        Memory<float> md;
        Memory<CUDAfloat> gpumd;
    
        CM Xp("Xp", np, d);
        cudaMemcpy((float *) Xp.data(), mxGetPr(XP_IN), sizeof(float)*np*d, cudaMemcpyHostToDevice);
        CM Xq("Xq", nq, d);
        cudaMemcpy((float *) Xq.data(), mxGetPr(XQ_IN), sizeof(float)*nq*d, cudaMemcpyHostToDevice);
        CM X("Xt", nt, d);
        cudaMemcpy((float *) X.data(), mxGetPr(XT_IN), sizeof(float)*nt*d, cudaMemcpyHostToDevice);

        // // auto ret = do_GF(Xp, Xpt, Xq, Xqt, step_size, start, end, maxiteration);
        auto ret = infer_KL(Xp, Xq, X, sigma_chosen, lambda_chosen);
    
        GRAD_OUT = mxCreateNumericMatrix(nt, d+1, mxSINGLE_CLASS, mxREAL);
        cudaMemcpy(mxGetPr(GRAD_OUT), (float *) ret.grad.data(), sizeof(float)*nt*(d+1), cudaMemcpyDeviceToHost);
        std::cout << ret.sigma << std::endl;
        SIGMA_OUT = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        float *psigmaOut = (float *) mxGetPr(SIGMA_OUT);
        *psigmaOut = ret.sigma;

        CuBLASErrorCheck(cublasDestroy(Matrix<CUDAfloat>::global_handle));
    }
}