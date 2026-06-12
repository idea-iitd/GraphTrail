from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
        name="pygcanl",
        packages=["pygcanl"],
        package_dir={"pygcanl": "."},
        ext_modules=[CppExtension("pygcanl._C", ["pygcanl.cpp"])],
        cmdclass={"build_ext": BuildExtension},
)
