from setuptools import setup, find_packages

setup(
    name="robustmip-bm",                       
    version="0.1.0",                        
    description="Benchmark extension for MMPose on ROBUST-MIPS",
    author="Zhe",
    install_requires=[
        "mmpose>=1.3.1"                   
    ],
    package_dir={"": "custom_src"},
    packages=find_packages("custom_src"),  
    # packages=find_packages(
    #     where="src",
    #     include=["surgicaltool_bm", "surgicaltool_bm.*"],
    # ),
    
)