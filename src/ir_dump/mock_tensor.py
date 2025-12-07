"""
MockTensor 类

用于创建模拟的 tensor 对象进行类型推断，不需要真实的 CUDA 设备
"""


class MockTensor:
    """模拟的 Tensor 对象，用于 cuda.tile kernel 的类型推断"""

    def __init__(self, shape, dtype="float32"):
        """
        创建 MockTensor

        Args:
            shape: tensor 形状，如 (512, 128)
            dtype_str: 数据类型，如 "float32", "float16", "int32"
        """
        self.shape = shape
        self.dtype_str = dtype
        self.device = "cuda"

        # 模拟 torch.dtype
        class MockDtype:
            def __init__(self, name):
                self.name = name

        self.dtype = MockDtype(dtype)
        self.data_ptr = lambda: 0  # 模拟指针

        # 实现 __cuda_array_interface__ 协议
        # 这是 CUDA 数组的标准接口
        dtype_map = {
            "float32": "<f4",
            "float64": "<f8",
            "float16": "<f2",
            "int32": "<i4",
            "int64": "<i8",
            "int16": "<i2",
            "int8": "<i1",
            "uint32": "<u4",
            "uint64": "<u8",
            "uint16": "<u2",
            "uint8": "<u1",
        }

        if dtype not in dtype_map:
            raise TypeError(
                f"Unsupported dtype: {dtype}. "
                "If this dtype is available in torch, consider using torch tensor with device='meta'."
            )

        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": dtype_map[dtype],
            "data": (0, False),  # (ptr, read_only)
            "version": 3,
        }
