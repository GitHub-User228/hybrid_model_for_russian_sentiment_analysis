from setuptools import find_packages, setup
from typing import List

# Get long description from the readme file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "1.0.2"

REPO_NAME = "hybrid_model_for_russian_sentiment_analysis"
AUTHOR_USER_NAME = "GitHub-User228"
SRC_REPO = "hybrid_model_for_russian_sentiment_analysis"
AUTHOR_EMAIL = "egor.udalov13@gmail.com"



def get_requirements(file_path: str) -> List[str]:
    """
    Retrieves the list of requirements from the specified file.

    Parameters:
    - file_path (str): Path to the requirements file.

    Returns:
    - List[str]: List of requirements.
    """
    with open(file_path, 'r') as file:
        # Exclude lines with '-e .' as it's commonly used to indicate editable installs
        requirements = [line.strip() for line in file if line.strip() != "-e ."]
    return requirements



setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Hybrid Model for detecting sensitive content in textual Russian news feeds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"" : "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=get_requirements('requirements.txt'),
    dependency_links=[
            'https://pypi.org/',
            'https://download.pytorch.org/whl/cu118'
        ],
    include_package_data = True
)