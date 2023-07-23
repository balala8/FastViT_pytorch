# FastViT_pytorch
This is an unofficial replication of FastViT using PyTorch. The research paper titled "FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization" is the source of this work. You can find the paper [FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/pdf/2303.14189.pdf)


# Description
There are some details in the paper that were not published, maybe I missed it. 
1. Activate the function The paper does not give which activation function to use, I used ReLU
2. In the first reparameterization block of the stem, the first and second convolutional branches utilize a stride of 2 (S=2) for downsampling, but the BatchNorm layers are not downsampling. However, their outputs are still added together. The same situation also occurs in the Patch Embedding process. In the replication, the BatchNorm layers have been removed.
3. In Figure 2, the number of Patch Embeddings block is 3, but in Table 2, the number of Patch Embeddings block is 4.


# Parameters And FLOPs
| Model       | Parameters (M) | FLOPs (G) | Image Size |
|-------------|---------------|-----------|------------|
| Official T8 | 3.6           | 0.7       | 256        |
| Replicate T8| 3.418         | 0.67      | 256        |
| Official T12| 6.8           | 1.4       | 256        |
| Replicate T12| 6.662        | 1.365     | 256        |
| Official S12| 8.8           | 1.8       | 256        |
| Replicate S12| 8.582        | 1.768     | 256        |
| Official SA12| 10.9         | 1.9       | 256        |
| Replicate SA12| 10.677      | 1.911     | 256        |
| Official SA24| 20.6         | 3.8       | 256        |
| Replicate SA24| 20.629      | 3.742     | 256        |
| Official SA36| 30.4         | 5.6       | 256        |
| Replicate SA36| 30.581      | 5.574     | 256        |
| Official MA36| 42.7         | 7.9       | 256        |
| Replicate MA36| 42.877      | 7.793     | 256        |

# Usage
Install PyTorch and FvCore (use this library in the paper to count the number of parameters and calculations)
```shell
pip install pytorch
pip install fcvore
```
just run model.py
```shell
python model.py
```

# TODO
Training on ImageNet1K is in progress. If anyone is willing to contribute by uploading and sharing their trained weights, it would be greatly appreciated. 
