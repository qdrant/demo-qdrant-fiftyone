# demo-qdrant-fiftyone

This repository contains a working example of integrating Qdrant with FiftyOne.
The example uses MNIST dataset and performs a classification of the test samples
using Approximate Neural Neighbours in the embedding space, using cosine distance.

## Running

The example requires some dependencies to be installed:

```shell
pip install -r requirements.txt
```

Qdrant has to be launched using Docker:

```shell
bash run_qdrant.sh
```

Finally, the script performing the classification and running FiftyOne in web
browser can be launched via:

```shell
python main.py
```
