# Vietnamese-wild-animal-recognition
A toy project for university's subject. The goal is to determine endangered animal in wild life. 
Thanks to:
- [YOLO-V12-CAM](https://github.com/rigvedrs/YOLO-V12-CAM) by rigvedrs – for class activation maps.
- [BioTrove](https://github.com/baskargroup/BioTrove) – for biodiversity-related visual modeling.
- Tran Trinh Hoang Chau and Tran Anh Minh for creating dataset and writing animal descriptions.

There are certain limitations in the system shown in the Validation set, such as:
- Overlapping animals (gaungua(0).jng)
- Environment blended animal (te_te_1(0).png)
- Animal with disproportional shape (ga_loi_lam_duoi_trang_0(1).png)

The Validation set has been scored on band 0 - 3:
- 0: The system fails to recognize the animal
- 1: The system detects the animal, but misclassifies it as a non-endangered animal
- 2: Correct label, but wrong heatmap
- 3: Correct label and heatmap

Limitations:

- Data properties: too small, even for few-shot learning. Furthermore, the range of animal poses is very narrow, limiting the model’s ability to generalize to real-world scenarios.

- Image preprocessing method: Images were manually cropped into almost square-shaped, then resized into (224, 224). This posed great challenge for animals with elongated bodies or protrusions since the less important details could be cropped out.

- Model performance: The fine-tuned CLIP model achieves:
  + Top-1 accuracy: ~80%
  + Top-3 accuracy: ~85%
  
  While reasonable for a toy project, this is still far from enough for real-world deployment.

- Weight limitations: The fine-tuned model weights exceed GitHub’s 100MB file size limit and therefore cannot be stored in this repository.
