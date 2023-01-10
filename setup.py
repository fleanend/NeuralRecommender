import setuptools
  
with open("README.md", "r") as fh:
    description = fh.read()
  
setuptools.setup(
    name="neuralrecommender",
    version="0.1.0",
    author="fleanend (Edoardo Ferrante)",
    author_email="edoardo@ferrante.ml",
    packages=["neuralrecommender"],
    description="Simple library for Recommendation Systems implemented with Neural Networks",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/fleanend/NeuralRecommender",
    license='MIT',
    python_requires='>=3.7',
    install_requires=["numpy","torch","typing-extensions"],
    tests_require=["colorama", "coverage", "green", "lxml", "unidecode"],
)
