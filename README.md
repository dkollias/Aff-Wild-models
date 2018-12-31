# Aff-Wild-models


The models on Aff-Wild can be downloaded [here](https://drive.google.com/open?id=1xkVK92XLZOgYlpaRpG_-WP0Elzg4ewpw).

The above link contains 3 folders named: "affwildnet-vggface-gru" , "affwildnet-resnet-gru" and "vggface".

The "vggface" folder contains two subfolders with 2 different models: both models are CNN networks based on VGG-FACE (with 3 fully connected layers with: i) 4096, 2000, 2 and ii) 4096, 4096,  2 units, respectively).

The "affwildnet-vggface-gru" folder contains the AffWildNet architecture (with no landmarks) as described in the paper entitled: ["Deep Affect Prediction in-the-wild: Aff-Wild Database and Challenge, Deep Architectures, and Beyond"](https://arxiv.org/pdf/1804.10938.pdf).

The "affwildnet-resnet-gru" folder contains the AffWildNet architecture (with no landmarks and no fully connected layer; a Resnet-50 followed by a GRU network) as described in the paper entitled: ["Deep Affect Prediction in-the-wild: Aff-Wild Database and Challenge, Deep Architectures, and Beyond"](https://arxiv.org/pdf/1804.10938.pdf).

Inside each of those folders, one can find the architectures of the networks, implemented in the Tensorflow environment and a readme explaining how to build/use them.
