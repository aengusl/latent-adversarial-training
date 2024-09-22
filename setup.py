from setuptools import setup, find_packages

setup(
    name="latent_at",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "wandb",
    ],
    author="goose",
    description="Latent Adversarial Training for LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)