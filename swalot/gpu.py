import torch
import GPUtil
import numpy as np
from rich import print
from functools import partial, update_wrapper


class MemoryProtector:
    def __init__(self, remain=256, device=0):
        self.device = device
        self.reserve_tensor = None
        self.remain = remain  # save x MB for other process, default: 256MB
        self.protecting_MB = 0

        # self.device_list = self._process_GPU_list(device)
        # self.smooth_mode = False

        self.protect()

    def protect(self):
        """
        Reserves (total free memory - self.remain) MB of memory on the GPU.
        """
        free_memory = GPUtil.getGPUs()[self.device].memoryFree
        reserve_MB = max(
            0, free_memory - self.remain
        )  # Reserve as much as possible, minus the remain

        if self.reserve_tensor is not None:
            # Free existing reserved memory before reallocating
            self.free_memory()

        # Convert MB to number of float32 elements (4 bytes each)
        reserve_elements = int(reserve_MB * 1024 * 1024 / 4)
        self.reserve_tensor = torch.from_numpy(
            np.empty(reserve_elements, dtype=np.float32)
        ).to(self.device)

        self.protecting_MB = reserve_MB
        print(
            "[bold cyan][Info][/bold cyan][bold magenta] | swat:[/bold magenta] Protecting RAM: [bold green]{} MB[/bold green]".format(
                reserve_MB
            )
        )

    def free_memory(self):
        self.reserve_tensor = None
        torch.cuda.empty_cache()

    def restore(self):
        self.protect()

    # def _process_GPU_list(self, GPU_list):
    #     """
    #     Process the input and convert all GPU identifiers to string format 'cuda:X'.

    #     :param GPU_list: The original GPU device list
    #     :return: A string list in the format 'cuda:X'
    #     """
    #     processed_list = []

    #     if isinstance(GPU_list, (int, str)):
    #         GPU_list = [GPU_list]

    #     for item in GPU_list:
    #         if isinstance(item, str) and item.startswith("cuda:"):
    #             processed_list.append(item)
    #         elif isinstance(item, (str, int)):
    #             processed_list.append(f"cuda:{int(item)}")
    #         else:
    #             raise ValueError("Items in GPU list should be either 'cuda:X' strings, 'X' strings or integers.")
    #     return processed_list

    # def allocate_tensor_on_GPU(self, index, *tensor_shape):
    #     """
    #     Allocate the tensor on the specified GPU.

    #     :param index: index in the GPU_list specifying the GPU on which to create the tensor
    #     :param tensor_shape: The shape of the tensor.
    #     :return: The created tensor
    #     """
    #     import torch
    #     device = self.GPU_list[index]
    #     tensor = torch.zeros(*tensor_shape, device=device)
    #     return tensor


class AutoMemoryProtector:
    def __init__(self, protector):
        self.protector = protector
        self.original_cuda = torch.Tensor.cuda

    def __enter__(self):
        """
        Case 1
        Monitor torch.Tensor.cuda() method
        """

        def custom_cuda(tensor, *args, **kwargs):
            try:
                return self.original_cuda(tensor, *args, **kwargs)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if self.protector.protecting_MB <= 0:
                        # print("It actually no memory...")
                        raise e
                    # print("Oops... Should be CUDA out of memory. But we have secret {} MB!".format(self.protector.protecting_MB))
                    self.protector.free_memory()
                    result = self.original_cuda(tensor, *args, **kwargs)
                    self.protector.restore()
                    return result
                else:
                    raise

        # Keep the custom_cuda function consistent with the original torch.Tensor.cuda method
        update_wrapper(custom_cuda, self.original_cuda)

        # Manually bind custom_cuda methods to the torch.Tensor class via custom_cuda.__get__(None, torch.Tensor)
        torch.Tensor.cuda = custom_cuda.__get__(None, torch.Tensor)

        """
        Case 2
        Monitor torch.Tensor.to(device) method
        """
        self.original_to = torch.Tensor.to

        def custom_to(tensor, *args, **kwargs):
            # Extract the device from args or kwargs
            device = kwargs.get("device", args[0] if args else None)
            if isinstance(device, str) and "cuda" in device:
                while True:
                    try:
                        return self.original_to(tensor, *args, **kwargs)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            if self.protector.protecting_MB <= 0:
                                raise e
                            self.protector.free_memory()
                            result = self.original_to(tensor, *args, **kwargs)
                            self.protector.restore()
                            return result
                        else:
                            raise
            else:
                return self.original_to(tensor, *args, **kwargs)

        update_wrapper(custom_to, self.original_to)
        torch.Tensor.to = custom_to.__get__(None, torch.Tensor)

        """
        Case 3
        Monitor functions like torch.rand(device=xxx) method, torch.zeros, torch.ones, etc.
        """
        self.original_functions = {}
        for func_name in ["rand", "zeros", "ones", "empty", "full", "randn"]:
            if hasattr(torch, func_name):
                original_func = getattr(torch, func_name)
                self.original_functions[func_name] = original_func

                def custom_func(*args, **kwargs):
                    while True:
                        try:
                            return self.original_functions[func_name](*args, **kwargs)
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                if self.protector.protecting_MB <= 0:
                                    raise e
                                self.protector.free_memory()
                                results = self.original_functions[func_name](
                                    *args, **kwargs
                                )
                                self.protector.restore()
                                return results
                            else:
                                raise

                setattr(torch, func_name, custom_func)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.Tensor.cuda = self.original_cuda
        torch.Tensor.to = self.original_to
        for name, func in self.original_functions.items():
            setattr(torch, name, func)
