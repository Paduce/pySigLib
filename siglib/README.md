<h1 align='center'>cpSIG and cuSIG</h1>
<h2 align='center'>C++ Libraries for Signature Computations on CPU and GPU</h2>

For high-dimensional data, most of the computational cost is incurred computing
the gram matrix

$$G_{i,j} = ( < \dot{x}_{s_i}, \dot{y}_{t_j} > )_{i,j}$$ 

which is done via matrix multiplication. Therefore, cpSIG and cuSIG take this matrix as
input, assuming it has already been computed via a performant linear algebra library. In pySigLib,
this is computed via `torch.bmm`.