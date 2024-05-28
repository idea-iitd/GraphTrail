from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
        name="pygcanl",
        ext_modules=[CppExtension('pygcanl',['pygcanl.cpp'])],
        cmdclass={'build_ext':BuildExtension},
)
