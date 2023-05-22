# PersonFormer - A Multimodal Video-based Personality Recognition System using Transformer

In this master’s thesis, we developed a multimodal personality recognition system using person image, face image, scene image, audio, text modalities and metadata for the Big 5 personality trait recognition.

## Description

We train the single modalities as own methods to see the effect on personality recognition of every single modalities and show different fusion strategies, network architectures, and transferlearning/retraining/training from scratch of the model backbones. We, therefore, split the project into methods M1 - M7, starting with single modalities in M1. 

Methods M2 and M3 contain variations of the fusion of the visual, audio, and textual stream. The methods in M4 show different late fusion variations of the trained methods M1 - M3 to reveal the influence of this fusion strategy on the performance of personality recognition. With the methods in M5, we describe the fusion of the visual modalities of person and face. We fuse the scene, audio, text, person, and face modality hierarchically in method M6 with different network settings by fusing the subnetworks from the previous methods M1, M2, and M3. The methods in M7 add metadata to the modalities, person, face, scene, text, and audio, from the previous methods, M1 and M5, on different levels and fuse the modalities hierarchically and on one level. 

Our best performing method with a mean average accuracy of 0.9124 is M7.6 with a one-level fusion of all modalities and metadata, followed by M7.2 with a hierarchical fusion of all modalities and a mean average accuracy of 0.9035.


## Getting Started
Here, we see our folder structure. The "Data_Analysis" folder contains the scripts for data analysis of the results. In "Data_Preprocessing" folder, we can find the scripts for frame extractionn for datatype 1 and datatype 2, and the person and face extraction script for the images. "Methods" folder contains of the scripts for the developed methods, architectures, dataloader and metric scripts, training/validation/test pipeline for training the methods and doing inference. We also offer tables as overview of the different methods.
```bash
.
└── personFormer/
    ├── Data_Analysis
    ├── Data_Preprocessing
    ├── Methods/
    │   ├── Baseline_Vision+Audio
    │   ├── Method_M1_Single_Modalities
    │   ├── Method_M2_M3_Vision+Audio+Text
    │   ├── Method_M4_Late_Fusion
    │   ├── Method_M5_Person_Face
    │   ├── Method_M6_Scene_Person_Face_Audio_Text
    │   └── Method_M7_Scene_Person_Face_Audio_Text_Metadata
    └── requirements.txt
```
### Dependencies
Please install the packages from the requirements file:

```bash
$ pip install -r requirements.txt
```

### Installing

* For training and evaluation, we use the First Impressions dataset for apparent personality recognition. Please download the data here: 
[First Impressions Dataset](https://chalearnlap.cvc.uab.cat/dataset/24/description/).
We need a certain arrangement of the data:
```bash
.
└── ChaLearn_First_Impression/
    ├── train/
    │   ├── train_video/
    │   ├── train_audio/
    │   ├── train_frames/
    │   ├── train_face_frames/
    │   ├── train_person_frames/
    │   ├── train_transcription/
    │   │   └── transcription_training.pkl
    │   ├── train_groundtruth/
    │   │   └── annotation_training.pkl
    │   └── eth_gender_annotations_train_val
    ├── val/
    │   ├── val_video/
    │   ├── val_audio/
    │   ├── val_frames/
    │   ├── val_face_frames/
    │   ├── val_person_frames/
    │   ├── val_transcription/
    │   │   └── transcription_validation.pkl
    │   ├── val_groundtruth/
    │   │   └── annotation_validation.pkl
    │   └── eth_gender_annotations_train_val
    └── test/
        ├── test_video/
        ├── test_audio/
        ├── test_frames/
        ├── test_face_frames/
        ├── test_person_frames/
        ├── test_transcription/
        │   └── transcription_test.pkl
        ├── test_groundtruth/
        │   └── annotation_test.pkl
        └── eth_gender_annotations_test
```
* We need the cropped face and person data and therefore, the next step is to preprocess the data using the scripts in "Data_Preprocessing".

* Our model backbones are R(2+1)D-34 <a id="1">[2]</a>, Video Swin Transformer <a id="1">[4]</a>, VGGish <a id="1">[3]</a>, and BERT <a id="1">[1]</a>. Please clone the repository of Video Swin Transformer and download relevant checkpoints of their pretrained models directly from their official page:
[Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) and add it in the respective folder of architectures, indicated with a "README.txt" file.

* We also did some experiments with pretrained model backbones to evaluate the effectness of transfer learning. Please download the weights of the pretrained VGGish on AudioSet from the source:
[VGGish weights](https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch/releases) and add it in the respective folder of checkpoint, indicated with a "README.txt" file.

* We provide our trained weights of our single modality methods M1.3 for person, M1.3 for scene, M1_14 for audio, M1_16 for text, and our final hierarchical personFormer model M7.2 and one-level personFormer model M7.6. These can be downloaded from:
[Released weights](https://drive.google.com/drive/folders/1MYuoeeFddnYZvVDidgxLdPWZbTKRK-pK?usp=sharing) 

### Executing program

* The training pipeline can be executed by, for example for audio, this command:
```
python train_audio.py --bs 4 --epochs 1 --root_dir_path C:/ChaLearn_First_Impression/
```
Default parameters: bs=16, epochs=100, lossfunction='mse', learningrate=1e-5, cudadevice='cuda', root_dir_path = './ChaLearn_First_Impression/

## Author

Jessica Kick


## Version History
* 0.1
    * Initial Release

## References
<a id="1">[1]</a> 
J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep
bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers) (J. Burstein, C. Doran, and T. Solorio, eds.), pp. 4171–4186, Association for Computational Linguistics, 2019.

<a id="1">[2]</a> 
D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, “A closer look at spatiotemporal convolutions for action recognition,” in 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pp. 6450–6459, Computer Vision Foundation / IEEE Computer Society, 2018.

<a id="1">[3]</a> 
S. Hershey, S. Chaudhuri, D. P. W. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal, D. Platt, R. A. Saurous, B. Seybold, M. Slaney, R. J. Weiss, and K. W. Wilson, “CNN architectures for large-scale audio classification,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017, New Orleans, LA, USA, March 5-9, 2017, pp. 131–135, IEEE, 2017.

<a id="1">[4]</a>
Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, “Video swin transformer,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 3192–3201, IEEE, 2022.

Pretrained weights and/or model backbone implementations are from:
R(2+1)D: https://github.com/moabitcoin/ig65m-pytorch
Video Swin Transformer: https://github.com/SwinTransformer/Video-Swin-Transformer
VGGish: https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch/releases
BERT: https://huggingface.co/docs/transformers/model_doc/bert
