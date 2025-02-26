from setuptools import setup, find_packages

setup(
    name="nothingbutnet",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "nothingbutnet": [
            "automation/models/*.json",
            "automation/hypotheses/*.json",
            "automation/prompts/*.txt"
        ]
    },
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "python-crontab>=2.5.0",
        "anthropic>=0.18.0",
        "kaggle>=1.5.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "predict-games=nothingbutnet.cli:main",
        ],
    },
) 