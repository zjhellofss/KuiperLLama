# 自制大模型推理框架

## 第三方依赖
1. google glog https://github.com/google/glog
2. google gtest https://github.com/google/googletest
3. sentencepiece https://github.com/google/sentencepiece
4. armadillo + openblas https://arma.sourceforge.net/download.html

**openblas作为armadillo的后端数学库，加速矩阵乘法等操作，也可以选用Intel-MKL**


## 模型下载地址
1. llama2 https://pan.baidu.com/s/1PF5KqvIvNFR8yDIY1HmTYA?pwd=ma8r

## 编译方法
```shell
  # 假设已经装好上述的第三方依赖
  mkdir build 
  cd build
  cmake ..
  make -j16
```

## 生成文本的方法
```shell
./llama_infer llama2_7b.bin tokenizer.model

```