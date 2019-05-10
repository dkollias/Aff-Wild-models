# Aff-Wild-models

## References

If you use any of the models/weights, please cite the following papers:

1.   D. Kollias, et. al.: "Deep Affect Prediction in-the-wild: Aff-Wild Database and Challenge, Deep Architectures, and Beyond". International  Journal  of  Computer  Vision  (2019).  

> @article{kollias2019deep,
  title={Deep affect prediction in-the-wild: Aff-wild database and challenge, deep architectures, and beyond},
  author={Kollias, Dimitrios and Tzirakis, Panagiotis and Nicolaou, Mihalis A and Papaioannou, Athanasios and Zhao, Guoying and Schuller, Bj{\"o}rn and Kotsia, Irene and Zafeiriou, Stefanos},
  journal={International Journal of Computer Vision},
  pages={1--23},
  year={2019},
  publisher={Springer}
}

2.  S. Zafeiriou, et. al. "Aff-Wild: Valence and Arousal in-the-wild Challenge", CVPRW, 2017.

>@inproceedings{zafeiriou2017aff, title={Aff-wild: Valence and arousal ‘in-the-wild’challenge}, author={Zafeiriou, Stefanos and Kollias, Dimitrios and Nicolaou, Mihalis A and Papaioannou, Athanasios and Zhao, Guoying and Kotsia, Irene}, booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on}, pages={1980--1987}, year={2017}, organization={IEEE} }

3. D. Kollias, et. al. "Recognition of affect in the wild using deep neural networks", CVPRW, 2017.

>@inproceedings{kollias2017recognition,
  title={Recognition of affect in the wild using deep neural networks},
  author={Kollias, Dimitrios and Nicolaou, Mihalis A and Kotsia, Irene and Zhao, Guoying and Zafeiriou, Stefanos},
  booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
  pages={1972--1979},
  year={2017},
  organization={IEEE}
}

## Pre-trained models:
The models on Aff-Wild can be downloaded from [here](https://drive.google.com/open?id=1xkVK92XLZOgYlpaRpG_-WP0Elzg4ewpw).


## Description:
The above link contains 3 folders named: "affwildnet-vggface-gru" , "affwildnet-resnet-gru" and "vggface".

The "vggface" folder contains two subfolders with 2 different models: both models are CNN networks based on VGG-FACE (with 3 fully connected layers with: i) 4096, 2000, 2 and ii) 4096, 4096,  2 units, respectively).

The "affwildnet-vggface-gru" folder contains the AffWildNet architecture (with no landmarks) as described in the paper entitled: ["Deep Affect Prediction in-the-wild: Aff-Wild Database and Challenge, Deep Architectures, and Beyond"](https://arxiv.org/pdf/1804.10938.pdf).

The "affwildnet-resnet-gru" folder contains the AffWildNet architecture (with no landmarks and no fully connected layer; a Resnet-50 followed by a GRU network) as described in the paper entitled: ["Deep Affect Prediction in-the-wild: Aff-Wild Database and Challenge, Deep Architectures, and Beyond"](https://arxiv.org/pdf/1804.10938.pdf).

Inside each of those folders, one can find the architectures of the networks, implemented in the Tensorflow environment and a readme explaining how to build/use them.
An evaluation file is also uploaded with detailed explanation inside.

The Aff-Wild database can be downloaded from [here](https://drive.google.com/file/d/1A6uU4XdO11o_VYSV5RCGiwCb9l4Wgk4s/view?usp=sharing).
Specific details about the database and related stuff can be read in the Challenge's [site](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/).

## Prerequisites:

- The code works with Tensorflow 1.8
- slim is also needed (it is incorporated within Tensorflow)

