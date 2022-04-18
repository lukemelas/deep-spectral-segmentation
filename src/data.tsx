// Project title
export const title = "Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization"

// Short version of the abstract
export const description = "We find that the Laplacian spectrum of the affinity matrices of self-supervised networks, particularly self-supervised vision transformers, can be used to decompose an image into meaningful semantic segments."

// Abstract
export const abstract = "Unsupervised localization and segmentation are long-standing computer vision challenges that involve decomposing an image into semantically-meaningful segments without any labeled data. These tasks are particularly interesting in an unsupervised setting due to the difficulty and cost of obtaining dense image annotations, but existing unsupervised approaches struggle with complex scenes containing multiple objects. Differently from existing methods, which are purely based on deep learning, we take inspiration from traditional spectral segmentation methods by reframing image decomposition as a graph partitioning problem. Specifically, we examine the eigenvectors of the Laplacian of a feature affinity matrix from self-supervised networks. We find that these eigenvectors already decompose an image into meaningful segments, and can be readily used to localize objects in a scene. Furthermore, by clustering the features associated with these segments across a dataset, we can obtain well-delineated, nameable regions, i.e.\ semantic segmentations. Experiments on complex datasets (Pascal VOC, MS-COCO) demonstrate that our simple spectral method outperforms the state-of-the-art in unsupervised localization and segmentation by a significant margin. Furthermore, our method can be readily used for a variety of complex image editing tasks, such as background removal and compositing."

// Institutions
export const institutions = {
  1: "Oxford University",
  // 2: "Oxford University"
}

// Authors
export const authors = [
  {
    'name': 'Luke Melas-Kyriazi',
    'institutions': [1],
    'url': "https://github.com/lukemelas/"
  },
  {
    'name': 'Christian Rupprecht',
    'institutions': [1],
    'url': "https://chrirupp.github.io/"
  },
  {
    'name': 'Iro Laina',
    'institutions': [1],
    'url': "http://campar.in.tum.de/Main/IroLaina"
  },
  {
    'name': 'Andrea Vedaldi',
    'institutions': [1],
    'url': "https://www.robots.ox.ac.uk/~vedaldi/"
  }
]

// Links
// TODO: Add the link to the paper
export const links = {
  'paper': "#", // "https://arxiv.org/abs/2002.00733",
  'github': "#", // "https://github.com/lukemelas/deep-spectral-segmentation"
  'demo': "#", // "https://huggingface.co/spaces/lukemelas/deep-spectral-segmentation"
  'poster': "/poster.pdf", // "https://huggingface.co/spaces/lukemelas/deep-spectral-segmentation"
}

// Acknowledgements
export const acknowledgements = "L. M. K. acknowledges the generous support of the Rhodes Trust. C. R. is supported by Innovate UK (project 71653) on behalf of UK Research and Innovation (UKRI) and by the European Research Council (ERC) IDIU-638009. I. L. and A. V. are supported by the VisualAI EPSRC programme grant (EP/T028572/1). "

// Citation
export const citationId = "melaskyriazi2022deep"
export const citationAuthors = "Luke Melas-Kyriazi and Christian Rupprecht and Iro Laina and Andrea Vedaldi"
export const citationYear = "2022"
export const citationBooktitle = "CVPR"

// Video
// TODO: Add a link to the video
export const video_url = "https://www.youtube.com/embed/ScMzIvxBSi4"