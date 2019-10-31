# MTP2019-ImageSuperResolution

The following work was done as part of the independent study under the supervision of Dr. Pawan Kumar. A list of references can be found at the end of this document.

### Introduction
Thorough this independent study, image super-resolution is attempted using Generative Adversarial Networks (GANs). The ML models have been created in Python using the fastai library. The models have been trained on dataset containing portraits of  humans and selfies. Hence they work reasonable well with similar kind of data. An accompanying web application has also been developed using python to allow users to enhance low quality images.

### Repository structure
A brief description of the repository structure is as follows:
- **Code**: This directory contains the code to create the ML models. The code has been written as a Jupyter Notebook which provides interaction as well as better reproducibility. This directory also contains the Web_App sub-dirrectory which contains the web application that can be used to enhance portrait images provided by the user.
- **Presentations**: This directory consists of the presentations about the Independent Study. Two sub-directories are present. One contains the initial introductory presentation and the other contains the final presentation. The presentations have been created in Latex using beamer.
- **Report**: This folder contains a detailed report about the work done during the independent study.
- **Ankit_Pant_Independent Study proposal.pdf**: This file contains the initial proposal about the independent study.
- **README.md**: This file contains a brief description of the project.
- **arch_srgan.png**: This file contains the diagrammatic representation of the model architecture.


### Running the Web Application
Running the web application requires the following steps:
1. Download/fork this repository
2. Ensure that docker is installed in the system
3. Make sure active internet connection is available during the first run of the web app (the required libraries and dependencies need to be downloaded)
4. Go inside the Code directory followed by going inside the Web_App sub-directory
5. Ensure docker service is running in your system
6. Open the terminal in the *Web_App* sub-directory and run the following command in the terminal:
   ```
   sudo docker build -t srwebapp . && sudo docker run --rm -it -p 5000:5000 srwebapp
7. If everything is running correctly, you'll see the following message  (among several others) in the terminal :
    > INFO: Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)

8. Go to your preferred web browser without closing the terminal
9. In the address bar type:
    ```http://localhost:5000/```
10. The web application is now running successfully and you can use the *Select Image* button to choose the image to enhance.
11. Clicking on *Enhance Image*  button would output the enhanced image.

### Running the Jupyter Notebook
The Jupyter notebook containg the code can be run locally using jupyter or can be run in platforms like Google Colab and Kaggle. However, the paths to the datasets will have to be changed.


### Architecture of the ML model
The ML model consists of three sub-models. The generator, the discriminator, and the up-sampler. All the three models are based on the models created in the fast.ai library. A diagrammatic  representation of the architecture can be in the seen in the *arch_srgan.png* file.

### Drawbacks of current implementation
- The algorithm current can output to a maximum resolution of 512∗512 pixels. It is due to lack of computational resources during  training.
- The algorithm only works for low  resolution images (preferably below 384 ∗ 384 pixels). Giving an image of larger resolution > (512 ∗ 512) will result in the output image having lower quality.
- The web application is not able to use the up-sampler based on VGG architecture. Hence it is not as accurate and can lead to over-softening of images which look  less sharp.
- Since the model was trained on a relatively small dataset (due to hardware resources), it may not output good quality output for images of certain scenes (e.g. overexposed scenes).


### Link to trained models
Since GitHub has an individual file size limit, they cannot be uploaded with the repository. However they are stored in Google Drive (along with datasets) and can be found in the following link: https://drive.google.com/drive/folders/1b0EjOKu_r9XXqMJKli6qxH8yTtaMp2X1?usp=sharing 


### References:
-   Goodfellow, Ian J. and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua, ``Generative Adversarial Nets'', arXiv e-prints, https://arxiv.org/pdf/1406.2661.pdf
- ``Generative Adversarial Network'', Wikipedia, the free encyclopedia, https://en.wikipedia.org/wiki/Generative\_adversarial\_network
- Alec Radford, Luke Metz, Soumith Chintala, ``Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks'', arXiv e-prints, https://arxiv.org/pdf/1511.06434.pdf
- Martin Arjovsky, Soumith Chintala, Léon Bottou, ``Wasserstein GAN'', arXiv e-prints, https://arxiv.org/pdf/1701.07875.pdf
- Min Lin, ``Softmax GAN'', arXiv e-prints, https://arxiv.org/pdf/1704.06191.pdf
-  Kevin Su, ``Introduction to Image Super-resolution'', http://www.cs.utsa.edu/~qitian/seminar/Fall04/superresolution/SR_slides_xsu.pdf 
- Jeremy Howard et al, ``fastai'', https://docs.fast.ai/, https://www.fast.ai/
- NVlabs, ``Flickr-Faces-HQ Dataset (FFHQ)'', https://github.com/NVlabs/ffhq-dataset
- Center for Research in Computer Vision, ``Selfie Data Set'', University of Central Florida https://www.crcv.ucf.edu/data/Selfie/
- Olaf Ronneberger, Philipp Fischer, Thomas Brox, ``U-Net: Convolutional Networks for Biomedical Image Segmentation'', arXiv e-prints, https://arxiv.org/pdf/1505.04597.pdf
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, ``Deep Residual Learning for Image Recognition'', arXiv e-prints,  https://arxiv.org/pdf/1512.03385.pdf
- ``Mean squared error'', Wikipedia, the free encyclopedia, https://en.wikipedia.org/wiki/Mean_squared_error
- ``Cross entropy'', Wikipedia, the free encyclopedia, https://en.wikipedia.org/wiki/Cross_entropy
- Neurohive, ``VGG16 – Convolutional Network for Classification and Detection'', https://neurohive.io/en/popular-networks/vgg16/
- ``Structural similarity'', Wikipedia, the free encyclopedia, https://en.wikipedia.org/wiki/Structural_similarity#Multi-Scale_SSIM
- ``Peak signal-to-noise ratio'', Wikipedia, the free encyclopedia, https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
- ``Least absolute deviations'', Wikipedia, the free encyclopedia, https://en.wikipedia.org/wiki/Least_absolute_deviations
- ``L1 norm'', Wikipedia, the free encyclopedia, https://en.wikipedia.org/w/index.php?title=L1_norm&redirect=no
- Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi, ``Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network'', arXiv e-prints, https://arxiv.org/pdf/1609.04802.pdf
- Mehdi S. M. Sajjadi, Bernhard Schölkopf, Michael Hirsch, ``EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis'', arXiv e-prints, https://arxiv.org/pdf/1612.07919.pdf










