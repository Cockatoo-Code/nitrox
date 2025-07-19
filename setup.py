from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nitrox",
    version="0.1.0",
    author="Cockatoo Team",
    author_email="zac@cockatoo.com",
    description="Deep, fast, clean. Media processing at maximum depth.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cockatoo-Code/nitrox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Zero dependencies - uses only stdlib and ffmpeg binary
    ],
    extras_require={
        "audio": [
            "numpy>=1.20.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "numpy>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nitrox=nitrox.cli:main",
        ],
    },
    keywords="video audio processing ffmpeg media conversion slicing",
    project_urls={
        "Bug Reports": "https://github.com/Cockatoo-Code/nitrox/issues",
        "Source": "https://github.com/Cockatoo-Code/nitrox",
        "Documentation": "https://github.com/Cockatoo-Code/nitrox#readme",
    },
) 