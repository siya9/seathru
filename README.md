# Underwater image enhancement using Sea-Thru algorithm on monocular images

Exploring underwater environments is crucial for underwater archaeology, marine resource research, and numerous other purposes. Images and videos are particularly important for underwater exploration due to their non-intrusive nature and ability to capture high-information content. However, light attenuation and back-scattering underwater often cause image distortion. When light enters the water, it gets refracted, absorbed, and scattered, causing images underwater to become dull and distorted. 

# How to run 

Enter the following command to execute the Python code code:

py seathru-mono.py --image input_folder --output output_folder

This is the link to the official <a href='https://github.com/nianticlabs/monodepth2/tree/d1c5f03c38305cae4e68917e472d2f9d4eda0b98'>
Monodepth2 repo
</a> that contains pretrained models that estimate the depthmap of images. 

The deep learning model compared against this Sea-thru enhancing technique can be obtained from the link: <a href='https://github.com/zhenqifu/PUIE-Net'>PUIE-Net</a>

The 2013 Aldabra dataset used in this project can be obtained from the following link: 
https://drive.google.com/file/d/1gClMHKolK7HsBc9nEOQAjLZclGWJ4sC7/view?usp=sharing
