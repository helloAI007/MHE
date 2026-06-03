#### 🔍MHE: Making the Classification Explanation Faithful to the Confidence Score

This is an official implementation for [Making the Classification Explanation Faithful to the Confidence Score](https://openaccess.thecvf.com/content/CVPR2026/html/Mi_Making_the_Classification_Explanation_Faithful_to_the_Confidence_Score_CVPR_2026_paper.html)(CVPR'26)

The MHE (Metropolis-Hastings Explainer), a black box explainer, provides explanations that are faithful to classification confidence. We also offer an enhanced version, MHE-pro. Additionally, MHE-e is tailored for tasks that involve explaining positive contributions areas separately.

#### Demo

To run the demo, you'll need to set up the environment first, then configure the image you want to input. Finally, just run it and you'll get the results.

##### Environment Setup

```bash
conda create -n mhe python=3.8 
conda activate mhe   
source setup.sh 
```

##### Configure

###### 1.Input Images

To get started, you'll need to point the script to your data. In `run.py`, set `imgs_dir` to the directory containing your input images, and `annotations_dir` to the folder with your annotation files (if you want to test PG/EBPG). Note that `annotations_dir` is only required for evaluating PG and EBPG—you can toggle this with the `is_PG` flag.

```python
# your input images directory
imgs_dir = 'xxx/'
```

###### 2.Other(optional)

The `run.py` script also offers several knobs to tweak:
- `saliency`: Set this to `True` if you want to save the final explanation map.
- `threshhold_ablation`, `terms_ablation`, and `size_ablation`: These control the acceptance threshold $\alpha$, the number of rounds, and the mask size, respectively.
- `abla_a` and `abla_b`: Correspond to the $a$ and $b$ thresholds used in the PNN metric (check the original paper for a deeper dive on PNN).

> In very rare scenarios, the sampling process may be slightly slower. however, a converged chain can be obtained through more iterations or independent repeated runs.

If you're using annotations, make sure the XML files under `annotations_dir` contain class bounding boxes and follow the structure shown below. (You can skip this parameter if `is_PG=False`.)

```
ImageNet/
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── annotations/
    ├── 1.xml
    ├── 2.xml
    └── ...
```

##### Run

```bash
# MHE
python run.py 1
# MHE-pro
python run.py 2
# MHE-e
python run.py 3
```

#### Citation

If you find this repository helpful, please consider citing our work and giving it a star 🌟

```latex
@InProceedings{Mi_2026_CVPR,
    author    = {Mi, Jian-Xun and Pan, Lu and Li, Weisheng},
    title     = {Making the Classification Explanation Faithful to the Confidence Score},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {38959-38968}
}
```

#### Acknowledgement

* RISE: https://github.com/eclique/RISE

Thanks for their wonderful works.

🛠 Open an issue for question or feedback.

