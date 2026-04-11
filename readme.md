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

### train_swinunetr.py

**SwinUNetR Architecture**

<img width="468" height="185" alt="image" src="https://github.com/user-attachments/assets/f1ee95ca-e3d8-4a04-9217-8d3804945359" />

Swin UNETR = Swin Transformer (Encoder) + U-Net-like Decoder (CNN-based).

It is a hybrid model that combines the strengths of:

- Transformers (excellent at capturing long-range/global context)
- CNNs (excellent at capturing local details and precise boundaries)

This makes it particularly strong for 3D medical segmentation tasks like BraTS brain tumor segmentation.

According to the diagram, the overall architecture flow is given below.

The model has two main parts:
- Left Side: Swin Transformer Encoder (Hierarchical Feature Extraction)
- Right Side: CNN Decoder (Feature Fusion + Upsampling to full resolution)

Input → 4-channel 3D volume (4, 128, 128, 128) (T1n, T1c, T2w, T2f)


**- Swin Transformer Encoder (The "Brain" of the Model)**

Patch Partition + Linear Embedding - The 3D volume is divided into small non-overlapping patches (usually 2×2×2 or 4×4×4). Each patch is flattened and linearly projected into a feature vector.

4 Hierarchical Stages (Stage 1 → Stage 4 and each stage does the following:

1. Swin Transformer Blocks (multiple blocks per stage)

2. Patch Merging (downsampling) → reduces spatial resolution by 2× and doubles the feature channels

3. Key Innovation: Shifted Window Multi-Head Self-Attention (SW-MSA)
Unlike standard Vision Transformer (which uses global self-attention and is very expensive), Swin uses:
- Window-based MSA (W-MSA): Attention computed only inside small local windows.
- Shifted Window MSA (SW-MSA): In alternate layers, the windows are shifted allows 	information to flow across windows.

This gives linear computational complexity while still modeling long-range dependencies effectively.

In your SWIN_CFG:

feature_size=48 → base number of channels
4 stages with increasing channels (48 → 96 → 192 → 384 typically)

- Output of Encoder: Multi-scale hidden features from all 4 stages (these are passed to the decoder via skip connections).

**- CNN Decoder (The "Reconstruction" Part)**

This part is similar to a classic U-Net decoder:
1. Takes the deepest feature map (from Stage 4) as the bottleneck.
2. Progressively upsamples the features back to the original resolution.
3. At each upsampling level, it concatenates (skip connection) the corresponding feature map from the Swin Transformer encoder.
4. Uses Residual Blocks (R-B) or simple convolution blocks to refine the features.
5. Final Segmentation Head: A small CNN that outputs 3 channels (WT, TC, ET).

**Some basic things but generally confusing.**

1. Difference Between Spatial Resolution and Feature Channels
- Spatial Resolution
  - What it means: The physical size of the feature map in 3D space (Depth × Height × Width)
  - In your input (128³ volume): 128 × 128 × 128
  - What it represents: How "fine-grained" the location information is
  - Typical change in encoder: Gets halved each stage

- Feature Channels (Feature Dimension / C)
  - What it means: The depth of the feature map — number of learned feature maps per voxel
  - In your input: Starts at 48 (in your config)
  - What it represents: Type of information stored (edges, textures, semantics, etc.)
  - Typical change in encoder: Gets doubled each stage
	
  Simple Analogy:

  Think of a 3D MRI volume as a Rubik’s cube.
  1. Spatial resolution = how many small cubes (voxels) the big cube is divided into (e.g., 128×128×128 small cubes).
  2. Feature channels = how much information you store inside each small cube (e.g., 48 different properties per cube: brightness, texture, tumor likelihood, etc.).
  3. Early layers → High spatial resolution + Low channels (lots of small cubes, each with basic info). 	
  4. Deep layers → Low spatial resolution + High channels (few big cubes, each with rich, abstract info).


2. What is "Patch Merging" Doing?
  
   In Swin UNETR (and the diagram you showed), Patch Merging is the   operation that happens between stages.

   How it works:
   1. It groups 2×2×2 neighboring patches (8 neighboring tokens/voxels).
   2. Concatenates their features → this temporarily increases channels by 8×.
   3. Then applies a linear layer to reduce it to exactly 2× the previous channel count.

   Result:
   1. Spatial resolution in each dimension is reduced by 2× (volume becomes 1/8th the size).
   2. Feature channels are doubled.

   This is repeated across the 4 stages of the Swin Transformer encoder.

3. Purpose of Reducing Spatial Resolution by 2× and Doubling Feature Channels.

   This is a standard hierarchical design used in almost all modern segmentation networks (U-Net, 	Swin UNETR, nnUNet, etc.). The goals are:

   Main Purposes:

   1. Increase Receptive Field Efficiently Deeper layers need to "see" a larger area of the original image. Reducing resolution by 2× effectively doubles the receptive field without using huge kernels or global attention (which is expensive).

   2. Shift from Local Details to Global Semantics
    
      -  Early stages (high resolution, low channels): Capture fine local details (edges, small tumor boundaries, texture).
      
      -  Later stages (low resolution, high channels): Capture high-level semantic information (what is tumor vs edema vs normal brain, overall shape, relationships between distant regions).

   3. Computational Efficiency Processing a feature map of size 64×64×64 with 96 channels is much cheaper than processing 128×128×128 with 48 channels, even though the total computation is balanced cleverly.
o	Multi-Scale Feature Learning This creates a feature pyramid. The decoder later uses skip connections from all stages to combine fine details (from early high-resolution maps) with rich context (from deep low-resolution maps).

4. How the Decoder Reverses the Process (Patch Expanding / Upsampling)

   The encoder in Swin UNETR progressively reduces spatial resolution (by 2×) and increases feature channels (by 2×) across 4 stages.

   The decoder does the opposite. It progressively increases spatial resolution back to the original size while reducing the number of channels, while intelligently fusing information.

   Step-by-step reversal process:

      a) Start from the Bottleneck
			The deepest encoder output (Stage 4) has the lowest spatial resolution (e.g., 16×16×16) but the richest features (highest number of channels, e.g., 384).

	b) Patch Expanding (Upsampling)
            Instead of simple bilinear/trilinear upsampling, Swin UNETR uses Patch Expanding (also called Patch Expanding Layer).
			It works almost like the reverse of Patch Merging:
			Takes a feature map of shape (C, D/2, H/2, W/2)
			Rearranges and projects it to (C/2, D, H, W) — effectively doubling the spatial resolution in each dimension while halving the channels. This is done using a linear layer + pixel-shuffle-like rearrangement.

   c) Skip Connection + Fusion
			At each decoder level, the upsampled feature is concatenated with the corresponding skip connection from the encoder at the same resolution.
			Example: After expanding from 32³ → 64³, it concatenates with encoder Stage 2 				output (which also has 64³ resolution).
			Then a Residual Block (R-B) or convolution block refines the combined features.

   d) Repeat for all levels
			This process is repeated 4 times until the feature map returns to the original input 				resolution (128×128×128).
		e) Final Segmentation Head
			A small 1×1×1 convolution layer maps the final high-resolution features to 3 output channels (WT, TC, ET).

5. Why This Hierarchical Design (Encoder-Decoder) is Better than a Flat (Single Resolution) Transformer for 3D Medical Data?

   Flat Transformer vs Hierarchical Swin UNETR (3D Medical Imaging):

   - Receptive Field
     - Flat Transformer (ViT-style): Global from the beginning (very expensive)
     - Hierarchical Swin UNETR: Gradually increases (efficient)
     - Winner: Hierarchical

   - Memory & Compute
     - Flat Transformer (ViT-style): Extremely high for 128³ (impractical)
     - Hierarchical Swin UNETR: Much lower (downsampling reduces tokens)
     - Winner: Hierarchical

   - Multi-scale Information
     - Flat Transformer (ViT-style): Only single scale
     - Hierarchical Swin UNETR: Rich multi-scale features (fine + coarse)
     - Winner: Hierarchical

   - Boundary Precision
     - Flat Transformer (ViT-style): Often blurry
     - Hierarchical Swin UNETR: Excellent (thanks to skip connections)
     - Winner: Hierarchical

   - Local Detail Capture
     - Flat Transformer (ViT-style): Weak (transformers struggle locally)
     - Hierarchical Swin UNETR: Strong (CNN decoder + high-res skips)
     - Winner: Hierarchical

   - Training Stability
     - Flat Transformer (ViT-style): Harder
     - Hierarchical Swin UNETR: Easier (better gradient flow via skips)
     - Winner: Hierarchical

   - Parameter Efficiency
     - Flat Transformer (ViT-style): Poor for high-res 3D
     - Hierarchical Swin UNETR: Very good
     - Winner: Hierarchical
	
	Key Advantages of Hierarchical Design for BraTS / 3D MRI:
   1.	Efficient Long-Range Context + Fine Details: Early encoder layers focus on local textures (edges of tumor). Deep encoder layers capture global context (tumor location relative to ventricles, overall shape). Decoder combines both → best of both worlds.
   2.	Dramatically Lower Memory: A flat transformer on 128³ volume would have ~2 million tokens → self-attention would be extremely slow and memory-heavy. Hierarchical design reduces tokens step-by-step (128³ → 64³ → 32³ → 16³), making attention feasible.
   3.	Better Boundary Segmentation: Tumors have sharp boundaries. Skip connections from high-resolution encoder features help the decoder recover precise edges that would otherwise be lost in deep layers.
   4.	Multi-Scale Feature Fusion: Medical images have objects at very different scales (small enhancing tumor + large edema). Hierarchical design naturally handles this.
   5.	Better Gradient Flow: Skip connections + residual blocks make training deeper networks much more stable.

Real-world Evidence:

- Pure Vision Transformers (flat) usually underperform U-Net style models on medical segmentation.
- Hybrid models like Swin UNETR, UNETR, and nnFormer (all hierarchical) consistently rank higher on BraTS leaderboards than flat transformers.

**Flow of the script and other things:**

Like nnUNet, it also imports all the basic functions from config.py.

**Important things to discuss:**

**Why DiceFocalLoss and sigmoid is used?**

DiceFocalLoss is a hybrid loss function provided by MONAI. It combines two popular losses:

- Dice Loss — Measures overlap (region similarity) between prediction and ground truth. Excellent for medical segmentation because it directly optimizes the Dice score (your main evaluation metric).
- Focal Loss — A variant of Binary Cross-Entropy that focuses on hard examples. It down-weights easy background voxels and heavily penalizes difficult (misclassified) tumor voxels.

	``` Loss = λ_dice * DiceLoss + λ_focal * FocalLoss ```

where, 
- LOSS_LAMBDA_DICE = 1.0
- LOSS_LAMBDA_FOCAL = 1.0
- FOCAL_GAMMA = 2.0 (the focusing parameter — higher gamma = more focus on 	hard examples)

We are not doing standard multi-class segmentation (where each voxel belongs to 	exactly one class). Instead, we convert the raw labels into 3 overlapping binary 	regions:
- Channel 0: Whole Tumor (WT) = NCR (1) + ED (2) + ET (3)
- Channel 1: Tumor Core (TC)  = NCR (1) + ET (3)
- Channel 2: Enhancing Tumor (ET) = ET (3) only

Important relationships:

1. ET is completely inside TC
2. TC is completely inside WT

So a single voxel can (and often does) belong to multiple regions at once. This is called multi-label segmentation (not multi-class).

#### Activation Functions Comparison (BraTS Segmentation)

- Softmax
  1. Behavior: Forces probabilities to sum to 1 (mutually exclusive)
  2. When to Use: Standard multi-class (one label per voxel)
  3. Suitable for BraTS: No
- Sigmoid
  1. Behavior: Independent probability per channel (0 to 1)
  2. When to Use: Multi-label / overlapping regions
  3. Suitable for BraTS: Yes

If you used softmax, the model would be forced to choose only one region per voxel → this would break the natural hierarchy (ET ⊂ TC ⊂ WT) and hurt performance badly. Sigmoid treats each output channel independently. Each channel gets its own probability (0.0 ~ 1.0). This perfectly matches the overlapping nature of WT/TC/ET.

Why this combination is popular for BraTS?

BraTS has extreme class imbalance (tumor voxels << background voxels) and due to this Dice Loss alone can be unstable or slow to converge. Focal Loss helps the model pay more attention to the tiny tumor regions (especially the Enhancing Tumor - ET). Together they give better convergence and higher final Dice scores than using either loss alone.

This loss is one of the go-to choices in recent BraTS papers and winning solutions.


**What is Automatic Mixed Precision Training?**

AMP is a technique that allows you to train deep neural networks using lower 	precision numbers (mostly float16 or bfloat16) instead of the default float32, while 	still maintaining accuracy. In our case, 3D volumes of shape (4, 128, 128, 128) are 	very memory hungry. Swin UNETR + gradient checkpointing still needs a lot of 	VRAM and hence, AMP helps us fit larger batch sizes or 	bigger models without running out of GPU memory.

There are two main components of AMP in PyTorch:
1.	Autocast -> This is the forward pass part of AMP. What it does is that it automatically casts operations to float16 (half precision) where it's safe and keeps some operations in float32 (e.g., reductions, softmax, loss computation) to avoid numerical instability. Due to this, much faster matrix multiplications on modern GPUs (A6000, RTX 40-series, etc.) can be done and significantly lower memory usage occurs during forward and backward pass.
2.	GradScaler or Gradient Scaler -> This is the backward pass safety mechanism. The problem it solves is that when using float16, gradients can become very small (underflow) → they become zero → model stops learning.
Solution is that GradScaler automatically scales up the loss before backward pass, then scales the gradients back down before optimizer step.

**Why Gradient Clipping is used?**

Especially with AMP + Transformers + DiceFocalLoss, gradients can become 	extremely large (exploding gradients) and problems like Loss becomes NaN, Model weights get destroyed, 	Training becomes unstable can occur. 
What gradient clipping does: It looks at the global norm of all gradients. If the norm > max_norm (you set 1	.0), it scales down all gradients proportionally so the total norm becomes exactly 1.0. This prevents exploding gradients while still allowing the model to learn. 

**What is sliding window inference and how it different from other inference 	methods?**

Sliding Window Inference is a technique used during inference (prediction) to 	process very large 3D medical volumes (like BraTS MRI scans) that cannot fit entirely into GPU memory at once.
Instead of feeding the entire 128×128×155 (or larger) volume to the model in one go, the method:
- Breaks the big volume into smaller overlapping patches (called ROIs — Regions of Interest).
- Runs the model on each small patch individually (or in small batches).

Why needed?

- Training → You train on small fixed-size patches, e.g., (128, 128, 128) with BATCH_SIZE=1. 
- Inference → The original BraTS volumes are often slightly different sizes and still too large to fit the whole volume + model activations on the GPU (especially for Swin UNETR or SegResNet).

	After inference, it stitches all the patch predictions back together into a full-volume 	output. We 	tried to run model(full_volume) and got Out of Memory (OOM) errors on A6000. Sliding window solves this by processing one small patch at a time while still using the full context the model was trained on.

	```bash
	logits = sliding_window_inference(
    	inputs        = images,           # shape: (1, 4, D, H, W)
   	    roi_size      = SW_ROI_SIZE,      # (128, 128, 128) ← same as training size
    	sw_batch_size = 1,                # how many patches processed at once
    	predictor     = model,            # your SwinUNETR or SegResNet
    	overlap       = SW_OVERLAP,       # 0.5 = 50% overlap
	    mode          = "gaussian",       # blending method
	)
	```

Why 50% overlap (SW_OVERLAP = 0.5)?

- Without overlap → visible seams/artifacts at patch boundaries.
- With overlap → smoother transitions and better accuracy, especially at tumor edges.

Gaussian blending (mode="gaussian"):
- Gives higher weight to predictions near the center of each patch.
- Reduces boundary artifacts significantly. This is the recommended setting for medical segmentation.

Other Inference Alternatives:

1. Standard/Common Alternatives:

- Full Volume Inference
  - How it Works: Feed the entire 3D volume to the model in one forward pass
  - Pros: Fastest, best global context, no seams
  - Cons: Extremely high memory usage (often OOM)
  - Best For: Small volumes or very large GPUs

- Tiling / Non-overlapping Patch Inference
  - How it Works: Split volume into non-overlapping tiles, predict each, then stitch
  - Pros: Simpler & faster than sliding window
  - Cons: Visible seams/artifacts at tile borders
  - Best For: Quick prototyping

- Coarse-to-Fine / Multi-Resolution
  - How it Works: First predict on downsampled low-res volume, then refine high-res patches
  - Pros: Good speed-accuracy trade-off
  - Cons: More complex pipeline
  - Best For: Large CT/MRI volumes

- Zoom-out / Zoom-in
  - How it Works: Global low-res prediction → focus on regions of interest for high-res
  - Pros: Efficient, focuses compute on important areas
  - Cons: Needs good coarse prediction
  - Best For: Foundation models (e.g., SegVol)

2. Advanced / Recent Methods (2024–2025)

- NMSW (No-More-Sliding-Window) — One of the most promising recent 	approaches (MICCAI 	2025). It uses a differentiable Top-K patch sampling module 	that learns to select only the most "important" patches instead of processing every overlapping patch uniformly. It combines 	selected high-res patch predictions with a coarse global (low-res) prediction. 
	Results: Up to 9–11× faster inference with similar or better accuracy and 91% less 	computation. It is model-agnostic (can be added to U-Net, Swin UNETR, etc.).
	(We are planning to use this in future with our custom hybrid model)
- Efficient Sliding Window Variants (used in nnUNet improvements) - Adaptive step size or skipping background-heavy regions. nnUNet sometimes optimizes the sliding window to reduce redundant computation on large background areas.
- Memory-Efficient Full-Volume Inference (via clever tiling with halo borders or out-	of-core techniques) - Process large volumes by carefully overlapping tiles with a "halo" (extra border) equal to the network's receptive field, then stitch exactly without artifacts.
- Patch Selection / Importance Sampling - During inference, predict a coarse map first, then only run high-resolution inference on patches with high uncertainty or 	high probability of containing the object (tumor).

3. Other Practical Approaches

	a) Test-Time Augmentation (TTA) combined with any method above (flip/rotate the volume or 	patches and average predictions).

	b) 2.5D Inference (your custom model direction): Process axial/coronal/sagittal slices or small 	stacks instead of full 3D volumes.
	c) Cascade Models: Low-resolution model for rough localization → high-resolution model only 	on candidate regions.

	d) Foundation Model Tricks (e.g., SegVol): Interactive or prompt-based inference with zoom 	mechanisms.
	
	
