# 2.5D Hybrid Custom Model for BraTS Dataset

A comprehensive deep learning pipeline for multi-region brain tumor segmentation on the BraTS 2023 GLI dataset using a 2.5D Hybrid Custom Model along with strong baseline architectures.

### Key Features

- Custom 2.5D Hybrid Model: Combines the efficiency of 2.5D processing with hybrid CNN-Transformer design for better spatial context and computational efficiency on 3D MRI volumes.
- Comparative Analysis: Rigorous benchmarking against multiple state-of-the-art models including SegResNet, Swin UNETR, and nnUNet v2.
- Multi-Region Segmentation: Accurate delineation of Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) regions.
- Detailed Evaluation: Comprehensive metrics including Dice Score, Hausdorff Distance (95%), ROC-AUC, sensitivity, specificity, and per-region performance with rich visualization plots.
- Explainable AI (XAI): Interpretable results using Class Activation Mapping (CAM), occlusion sensitivity, and modality importance analysis to understand model decisions on MRI sequences (T1n, T1c, T2w, T2f).

## Basic workflow

config.py -> dataset.py -> train_nnUNet.py -> train_swin_unetr.py -> train_segResNet.py -> train_custom.py -> evaluate.py -> xai.py              

## What each file does?
### config.py

This file contains all the values and sizes like spatial_size, batch_size etc. which is used by different models in future. Any change in size or something will affect the whole pipeline. This acts as Single Source of Truth File for all the models in future. BRATS dataset has its own structure where the main data folder has subfolders whose filename is unique patient id and inside every patient folder there are 5 files representing 4 modalities which are as follows:

1. Channel 0 → T1 native (anatomy)
2. Channel 1 → T1 contrast (enhancing tumor)
3. Channel 2 → T2 weighted
4. Channel 3 → T2 FLAIR (edema)

This is a specific order accepted by nnUNet model. If this order is messed, dice value will drop by 0.4-0.6 value.
Though there are 4 unique labels (0,1,2,3) in BRATS dataset:

1. 0 = background
2. 1 = necrotic tumor core (NCR)
3. 2 = peritumoral edematous / invaded tissue (ED)
4. 3 = GD-enhancing tumor (ET)

and official BRATS dataset is divided into 3 regions as follows:

1. WT (Whole Tumor) = label 1+ label 2+ label 3 = output channel 0
2. TC (Tumor Core) = label 1+ label 3 = output channel 1
3. ET (Enhancing Tumor) = label 3 = output channel 2


This region order is also fixed. We convert the raw integer mask to a 3-channel binary mask in dataset.py. Models predict 3 segmentation channels. Dice is computed per channel.

Fixed the output folder and inside main output folder, and all the models can create their own folder and store their outputs.

In this file, we have fixed seed value and divided the whole data into (70-15-15) train-val-test split. All the files under train, val and test data is mentioned in splits.json, which is also created in this file only.

BRATS dataset volume has a lot of free space and huge background which can be either air or skull and if we will normalize all the pixels, zero will dominate and it will lead to bad mean and standard deviation and ultimately bad dice value. Thus, we only normalize the pixels of brain.

As we are training only on a subset of dataset (150 patients out of 950), the data is highly imbalanced due to split, and in to compensate that we are using DiceFocalLoss and Cosine Annealing for good convergence. Early stopping is also applied on 20 Non-Improvement Epochs.

All the model’s basic architecture and configuration related to Explainability AI is also present in this file only. (Will explain architecture and explainability AI later when we will talk about their standalone file).

We also used dynamic learning rate starting from 1e-4 with a weight decay of 1e-5 and can go to minimum of 1e-6.

### dataset.py

This is data foundation file which will handle preprocessing of data for all the models. It imports necessary functions from config.py.

First, it ensures that the main root directory where all the patient folders are stored. It checks whether all the patient folder contains all 4 modalities and 1 segmentation file and if any case missed any of the file, it would skip that patient. 

After that it will create splits.json, If not created, else it will import it from the disk only. As it has a fixed seed value, train-val-test split will not change in long run. Then functions like load_nifti loads .nii.gz file and passes to load_case in which they force a particular order of image and label where order of channels in image is [t1n, t1c, t2w, t2f] and float32 array of shape in image is (4, D, H, W). Similarly in labels, raw integer labels are in form of [0, 1, 2, 3] and int8 array of shape (D, H, W). 

Going forward, we will normalize only non-zero voxels by Z-Score because mean and standard deviation calculated over non-zero voxels will exclude the large zero background, otherwise if we will apply normalization on whole channel, large zero background will make mean and standard deviation biased towards zero. This is one of our learning in this journey.

Conversion of labels to 3-channel binary mask is also necessary. If image is small, pad_or_crop function will pad it with zeros and if its larger, then just simply perform center-crop. Channels and everything will remain same except the spatial dimensions which will be now equal to target size. Performing lightweight augmentation including random flip and random intensity scale where each channel is multiplied by small scaler. 

Data loaders of train, val and test will also be created in this file only. Final image dimensions will be [4, 128, 128, 128] and label dimensions will be [3, 128, 128, 128].

### train_nnunet.py
nnUNet is one of the strongest model earlier used in research and often used as baseline in research nowadays and it controls everything from preprocessing, naming, training etc.

#### Architecture Discussion:

nnUNet does not invent a completely new neural network architecture. Its name literally means "No New U-Net" and it takes the classic U-Net and makes it extremely powerful by automatically configuring everything around it.

At its heart, it uses 3D U-Net as shown below. 
<img width="468" height="235" alt="image" src="https://github.com/user-attachments/assets/d335e91f-2d9c-4d93-b769-5dce7f449efd" />


A 1×1×1 convolution that maps to the number of output channels. In our case,  nnUNet uses region-based training where it predicts overlapping binary masks for WT, TC and ET.

Skip Connections are the magic of U-Net as they allow high-resolution details from the encoder to flow directly to the decoder, which is crucial for precise tumor boundary segmentation.

A slight difference which is that in original UNet structure, we use ReLU whereas in nnUNet we use Leaky ReLU.
<img width="468" height="225" alt="image" src="https://github.com/user-attachments/assets/e47eeaab-cc55-4406-9d8f-0baeee463ddd" />

 
For 3D datasets like BraTS, nnUNet typically plans these:
- 3d_fullres: The main one we are using, full resolution 3D U-Net.
- 3d_lowres + 3d_cascade_fullres: A two-stage cascade (means going from low-res coarse prediction to full-res refinement). Often skipped for BraTS because the fullres patch already covers enough context.
- 2d: Slice-by-slice 2D U-Net.

In our pipeline, we are training only 3d_fullres on fold 0.

#### High Level Flow:

- Step 1: Convert to nnUNet raw format (imagesTr/labelsTr/imagesTs)
- Step 2: Plan + Preprocess (fingerprint + experiment planning)
- Step 3: Train (with live tqdm + custom trainer)
- Step 4: Export best checkpoint + plans to your unified CHECKPOINT_DIR\

It reuses the split we created inside dataset.py and uses its own approach to train the model. It creates three environment variables namely:
1.	nnUNet_raw -> It keeps the original raw dataset here in its custom format which it accepts (tr -> training and ts -> testing).
2.	nnUNet_preprocessed -> nnUNet preprocess the dataset on its own. It applies best patch size, batch size, data augmentation, resampling, normalization, cropping etc. and stores the information in nnUNetPlans.json
3.	nnUNet_results -> This is where nnUNet stores all the results like best and latest checkpoints, train and val loss graph, logs etc.

Originally nnUNet is imported from MONAI library and trained on 5 folds and 1000 epochs but here with wrapper we are training it on only one fold (0th fold) with 140 epochs.

We force the same modalities, labels, dataset split, regions here which we introduced in config.py and dataset.py file. All the training is done by running shell commands. 

We can continue the training by extracting the epochs from logs which we stored.

Careful handling of best checkpoints and logs is necessary because these things will be used later in evaluation (evaluate.py) and explainability ai (xai.py)

