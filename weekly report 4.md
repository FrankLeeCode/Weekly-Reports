# Weekly Report 4

**Oct.28** 

## Paper Reading

### MSDN

MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning

#### motivation

- 之前的工作只是简单的对其全局特征与其相关联的类别语义向量，或者使用单向的注意力去学道有限的潜在寓意表示。这些方法都不能发掘图片和属性特征间的内在语义知识。



#### datasets

| name                             | Object | Category | attributes | image_count | link                                                         |
| -------------------------------- | ------ | -------- | ---------- | ----------- | ------------------------------------------------------------ |
| CUB (Caltech UCSD Birds 200)     | Bird   | 200      | 322        | 11,788      |                                                              |
| SUN (SUN Attribute)              | Scene  | 717      | 102        | 14,340      | [SUN Attribute Database](https://cs.brown.edu/~gmpatter/sunattributes.html) |
| AWA2 (Animals with Attributes 2) | Animal | 50       | 85         | 37,322      | [animals with attributes 2](https://cvml.ist.ac.at/AwA2/#)   |

#### innovation

使用两个子网络：attribute→visual attention 和 visual→attribute attention 以达到双向注意力机制。更好的学习到

## Question..?
