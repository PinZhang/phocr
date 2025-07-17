
## Transformer在onnxruntime中的推理优化

1. 保持最简单的onnxruntime依赖不再包含第三方依赖库
2. 尽可能提升推理性能

## 网络结构
Transformer有Encoder和Decoder构成，Encoder每次推理只运行一次，Decoder每解码1个step都要
运行1次。 Encoder输出作为Decoder解码Cross Attention的输入。

## 优化点说明：

1. cross attention中k, v的投影计算从Decoder中移出，逻辑加入到Encoder中，Encoder的输出变为cross_attn_kvs
`(num_layers, batch_size, head_size, src_len, 2 * head_dim)`

2. 在beam search中, 通常会将cross_attn_kvs沿batch_size维度做repeat，这样会显著增加计算的显存，修改Attention计算方式
改为延迟广播计算，输入给Decoder的cross_attn_kvs不再进行repeat操作

3. 在beam search中，通常会有两次topk操作，第一次每条beam进行topk候选，第二次所有beam进行topk候选。
第一次topk如果在cpu中计算是非常耗时的，将第一次topk计算操作移入到decoder计算逻辑中。在外层的beam search控制逻辑中只进行第二次topk操作。

4. 所有的self kv cache在Decorder中合并为merged_self_attn_kvs，由Decorder输出

5. 使用onnxruntime io bindding方式进行推理计算，输入，输出如非必要保持在最近的计算设备上，例如merged_self_attn_kvs

6. Optional. onnxruntime 自动显存分配复用性非常一般，使用torch提前分配一块在cuda上大小为
`(num_layers, batchxbeam, head_num, max_seq_len, 2 * head_dim)` kv cache空间。 每次推理都在这块空间上进行分配和io binding使用。（注意事项: io binding 不能绑定cudaMalloc的裸指针）


## 后续改进

1. 当前实现依赖两个session独立进行encoder和decorder计算，可以实现动态 loop或者custom op等方式进行合并，减少中间显式变量传递。
