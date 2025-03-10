**multi-dimension container, for Numerical calculations**

**+-*/ element-wise calculation speed compara**




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
|                         | mkl(intel cpu)                   | 5476ms                     |
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
|                         | mkl(intel cpu)                   | 5301ms                     |
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
|                         | mkl(intel cpu)                   | 6347ms                     |
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
|                         | mkl(intel cpu)                   | 3516ms                     |
|                         | eigen matrix                     | 3411ms                     |
|                         | eigen vector                     | 3408ms                     |



**1d-10M-data, 100 times**

float:   avx2 = normal = eigen = mkl

double:  mkl >= eigen >= avx = normal
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 3090ms                     |
|                         | avx2                             | 3076ms                     |
|                         | mkl(intel cpu)                   | 3046ms                     |
|                         | eigen matrix                     | 3118ms                     |
|                         | eigen vector                     | 3113ms                     |
| double                  | norm                             | 6642ms                     |
|                         | avx2                             | 6589ms                     |
|                         | mkl(intel cpu)                   | 6233ms                     |
|                         | eigen matrix                     | 6394ms                     |
|                         | eigen vector                     | 6481ms                     |


**2d-3*3-data, 100000000 times**

float:   avx2 >> normal = vector > eigen = mkl

double:  normal = vector > eigen > avx2(???) > mkl
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1310ms                     |
|                         | vector                           | 1315ms                     |
|                         | avx2                             |  322ms                     |
|                         | mkl(intel cpu)                   | 8013ms                     |
|                         | eigen matrix                     | 1802ms                     |
| double                  | norm                             | 1305ms                     |
|                         | vector                           | 1318ms                     |
|                         | avx2                             | 3982ms                     |
|                         | mkl(intel cpu)                   | 7306ms                     |
|                         | eigen matrix                     | 2023ms                     |


**2d-10*10-data, 5000000 times**

float:   avx2 >> eigen > mkl > normal = vector

double:  avx2 > eigen > mkl > normal = vector
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1274ms                     |
|                         | vector                           | 1329ms                     |
|                         | avx2                             |  111ms                     |
|                         | mkl(intel cpu)                   |  800ms                     |
|                         | eigen matrix                     |  422ms                     |
| double                  | norm                             | 1484ms                     |
|                         | vector                           | 1500ms                     |
|                         | avx2                             |  449ms                     |
|                         | mkl(intel cpu)                   | 1150ms                     |
|                         | eigen matrix                     |  712ms                     |



**2d-30*30-data, 1000000 times**

float:   avx2 > eigen > mkl > normal = vector

double:  avx2 > eigen > mkl > normal = vector
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1067ms                     |
|                         | vector                           | 1092ms                     |
|                         | avx2                             |  339ms                     |
|                         | mkl(intel cpu)                   |  577ms                     |
|                         | eigen matrix                     |  516ms                     |
| double                  | norm                             | 2408ms                     |
|                         | vector                           | 2414ms                     |
|                         | avx2                             |  919ms                     |
|                         | mkl(intel cpu)                   | 1380ms                     |
|                         | eigen matrix                     | 1204ms                     |



**2d-300*300-data, 20000 times**

float:   avx2 > eigen > mkl > normal = vector

double:  avx2 > eigen > normal = vector = mkl
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1495ms                     |
|                         | vector                           | 1521ms                     |
|                         | avx2                             | 1030ms                     |
|                         | mkl(intel cpu)                   | 1428ms                     |
|                         | eigen matrix                     | 1269ms                     |
| double                  | norm                             | 5625ms                     |
|                         | vector                           | 5627ms                     |
|                         | avx2                             | 5393ms                     |
|                         | mkl(intel cpu)                   | 5655ms                     |
|                         | eigen matrix                     | 5424ms                     |



**2d-1000*1000-data, 1000 times**

float:   avx2 = eigen = mkl = normal = vector

double:  avx2 = eigen = normal = vector = mkl
| Type                    | Method                           | cost time                  |
| :---------------------- | :------------------------------- |--------------------------- |
| float                   | norm                             | 1603ms                     |
|                         | vector                           | 1556ms                     |
|                         | avx2                             | 1523ms                     |
|                         | mkl(intel cpu)                   | 1644ms                     |
|                         | eigen matrix                     | 1618ms                     |
| double                  | norm                             | 3520ms                     |
|                         | vector                           | 3494ms                     |
|                         | avx2                             | 3394ms                     |
|                         | mkl(intel cpu)                   | 3484ms                     |
|                         | eigen matrix                     | 3425ms                     |