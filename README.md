multi-dimension container, for Numerical calculations

+-*/ element-wise calculation speed conpara

**1d-10-data, 50000000 times**

avx2 >> eigen vector > eigen matrix > normal >> mkl
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1614ms                     |
|                         | avx2                             |  310ms                     |
|                         | mkl(intel cpu)                   | 5764ms                     |
|                         | eigen matrix                     | 1450ms                     |
|                         | eigen vector                     | 1225ms                     |
| double                  | norm                             | 1789ms                     |
|                         | avx2                             |  432ms                     |
|                         | mkl                              | 5476ms                     |
|                         | eigen matrix                     | 1593ms                     |
|                         | eigen vector                     | 1397ms                     |


**1d-300-data, 10000000 times**

avx2 > normal >= eigen > mkl
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1841ms                     |
|                         | avx2                             | 1268ms                     |
|                         | mkl(intel cpu)                   | 2536ms                     |
|                         | eigen matrix                     | 1837ms                     |
|                         | eigen vector                     | 1844ms                     |
| double                  | norm                             | 3907ms                     |
|                         | avx2                             | 3068ms                     |
|                         | mkl                              | 5301ms                     |
|                         | eigen matrix                     | 4189ms                     |
|                         | eigen vector                     | 4245ms                     |


**1d-10K-data, 300000 times**

avx2 > eigen = normal > mkl
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 2239ms                     |
|                         | avx2                             | 1722ms                     |
|                         | mkl(intel cpu)                   | 2360ms                     |
|                         | eigen matrix                     | 2180ms                     |
|                         | eigen vector                     | 2147ms                     |
| double                  | norm                             | 4724ms                     |
|                         | avx2                             | 4166ms                     |
|                         | mkl                              | 6347ms                     |
|                         | eigen matrix                     | 4969ms                     |
|                         | eigen vector                     | 5065ms                     |

**1d-300K-data, 4000 times**

avx2 = normal = eigen = mkl
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1730ms                     |
|                         | avx2                             | 1696ms                     |
|                         | mkl(intel cpu)                   | 1714ms                     |
|                         | eigen matrix                     | 1704ms                     |
|                         | eigen vector                     | 1703ms                     |
| double                  | norm                             | 3388ms                     |
|                         | avx2                             | 3397ms                     |
|                         | mkl                              | 3516ms                     |
|                         | eigen matrix                     | 3411ms                     |
|                         | eigen vector                     | 3408ms                     |

**1d-10M-data, 100 times**

float:   avx2 = normal = eigen = mkl

double:  mkl >= eigen >= avx = normal
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 3090ms                     |
|                         | avx2                             | 3076ms                     |
|                         | mkl                              | 3046ms                     |
|                         | eigen matrix                     | 3118ms                     |
|                         | eigen vector                     | 3113ms                     |
| double                  | norm                             | 6642ms                     |
|                         | avx2                             | 6589ms                     |
|                         | mkl(intel cpu)                   | 6233ms                     |
|                         | eigen matrix                     | 6394ms                     |
|                         | eigen vector                     | 6481ms                     |
