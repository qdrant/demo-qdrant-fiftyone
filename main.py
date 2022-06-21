import collections

import fiftyone as fo
import fiftyone.zoo as foz
import qdrant_client as qc

from tqdm import tqdm
from qdrant_client.http.models import Distance


# Load the data
dataset: fo.Dataset = foz.load_zoo_dataset("mnist", max_samples=2500)
train_dataset: fo.Dataset = dataset.match_tags(tags=["train"])

# Compute embeddings
model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
train_embeddings = train_dataset.compute_embeddings(model)
train_ground_truth = [
    {"ground_truth": sample.ground_truth.label} for sample in train_dataset
]

# Load the train embeddings into Qdrant
client = qc.QdrantClient(host="localhost")
client.recreate_collection(
    collection_name="mnist",
    vector_size=train_embeddings.shape[1],
    distance=Distance.COSINE,
)
client.upload_collection(
    collection_name="mnist",
    vectors=train_embeddings,
    payload=train_ground_truth,
)

# Assign the labels to test embeddings by selecting the most common label
# among the neighbours of each sample
test_dataset: fo.Dataset = dataset.match_tags(tags=["test"])
test_embeddings = test_dataset.compute_embeddings(model)

# Call Qdrant to find the closest data points
for sample, embedding in tqdm(zip(test_dataset.view(), test_embeddings)):
    search_results = client.search(
        collection_name="mnist",
        query_vector=embedding,
        with_payload=True,
        top=15,
    )
    # Count the occurrences of each class and select the most common label
    # with the confidence estimated as the number of occurrences of the most
    # common label divided by a total number of results.
    counter = collections.Counter(
        [point.payload["ground_truth"] for point in search_results]
    )
    predicted_class, occurences_num = counter.most_common(1)[0]
    confidence = occurences_num / sum(counter.values())
    sample["ann_prediction"] = fo.Classification(
        label=predicted_class, confidence=confidence
    )
    sample.save()

# Evaluate the ANN predictions, with respect to the values in ground_truth
results: fo.ClassificationResults = test_dataset.evaluate_classifications(
    "ann_prediction", gt_field="ground_truth", eval_key="eval_simple"
)

# Display the classification metrics
results.print_report()

if __name__ == "__main__":
    # Display FiftyOne app, but include only the wrong predictions
    session = fo.launch_app(dataset.match_tags("test"))
    session.view = test_dataset.match(fo.ViewField("eval_simple") == False)
    session.wait()
