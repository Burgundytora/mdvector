multi-dimension container, for Numerical calculations

**1d-short-data calculation speed compara**
300 element, +-*/ element-wise calculation,  run 10000000 times
| Type          | Method                 | cost time             |
| :------------ | :--------------------- |---------------------- |
| float         | norm                   | 1841ms                |
|               | avx2                   | 1268ms                |
|               | mkl                    | 2536ms                |
|               | eigen matrix           | 1837ms                |
|               | eigen vector           | 1844ms                |
| double        | norm                   | 3907ms                |
|               | avx2                   | 3068ms                |
|               | mkl                    | 5301ms                |
|               | eigen matrix           | 4189ms                |
|               | eigen vector           | 4245ms                |
