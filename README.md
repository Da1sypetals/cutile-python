# cuTile 类型提示

可参考`src/matmul_lsp.cutile.py`

1。编写cutile kernel
2。文件后缀改为.cutile.py
3。参考`matmul_lsp.cutile.py`编写一个boilerplate，包括用MockTensor输入的tensor，指定kernel的传参；调用`get_kernel_shapes_info`并且将得到的内容以json输出到stdout。stdout不要输出其他内容。
4。启动cutile-typeviz扩展，可以看到类型提示。