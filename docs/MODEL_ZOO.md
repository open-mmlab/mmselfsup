# Model Zoo

## Pre-trained model download links

<table>
   <thead>
      <tr>
         <th>Method</th>
         <th>Config</th>
         <th>Remarks</th>
         <th>Download link</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td>
         <td>-</td>
         <td>torchvision</td>
         <td><a href="https://drive.google.com/file/d/11xA3TOcbD0qOrwpBfYonEDeseE1wMfBh/view?usp=sharing" target="_blank" rel="noopener noreferrer">imagenet_r50-21352794.pth</a></td>
      </tr>
      <tr>
         <td>Random</td>
         <td>-</td>
         <td>kaiming</td>
         <td><a href="https://drive.google.com/file/d/1UaFTjd6sbKkZEE-f58Zv30bnx7C1qJBb/view?usp=sharing" target="_blank" rel="noopener noreferrer">random_r50-5d0fa71b.pth</a></td>
      </tr>
      <tr>
         <td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td>
         <td>selfsup/relative_loc/r50.py</td>
         <td>default</td>
         <td></td>
      </tr>
      <tr>
         <td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td>
         <td>selfsup/rotation_pred/r50.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1t3oClmIvQ0p8RZ0V5yvQFltzjqBO823Y/view?usp=sharing" target="_blank" rel="noopener noreferrer">rotation_r50-cfab8ebb.pth</a></td>
      </tr>
      <tr>
         <td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td>
         <td>selfsup/deepcluster/r50.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1GxgP7pI18JtFxDIC0hnHOanvUYajoLlg/view?usp=sharing" target="_blank" rel="noopener noreferrer">deepcluster_r50-bb8681e2.pth</a></td>
      </tr>
      <tr>
         <td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td>
         <td>selfsup/npid/r50.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1sm6I3Y5XnCWdbmeLSF4YupUtPe5nRQMI/view?usp=sharing" target="_blank" rel="noopener noreferrer">npid_r50-dec3df0c.pth</a></td>
      </tr>
     <tr>
         <td></td>
         <td>selfsup/npid/r50_ensure_neg.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1FldDrb6kzF3CZ7737mwCXVI6HE2aCSaF/view?usp=sharing" target="_blank" rel="noopener noreferrer">npid_r50_ensure_neg-ce09b7ae.pth</a></td>
      </tr>
      <tr>
         <td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td>
         <td>selfsup/odc/r50_v1.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1EdhJeZAyMsD_pEW7uMhLzos5xZLdariN/view?usp=sharing" target="_blank" rel="noopener noreferrer">odc_r50_v1-5af5dd0c.pth</a></td>
      </tr>
      <tr>
         <td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td>
         <td>selfsup/moco/r50_v1.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1ANXfnoT8yBQQBBqR_kQLQorK20l65KMy/view?usp=sharing" target="_blank" rel="noopener noreferrer">moco_r50_v1-4ad89b5c.pth</a></td>
      </tr>
      <tr>
         <td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td>
         <td>selfsup/moco/r50_v2.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1Cc5qMjPkKW6WeM4ic9Tq-IBxswyJhMnF/view?usp=sharing" target="_blank" rel="noopener noreferrer">moco_r50_v2-58f10cfe.pth</a></td>
      </tr>
      <tr>
         <td></td>
         <td>selfsup/moco/r50_v2_simclr_neck.py</td>
         <td>-&gt; SimCLR neck<br></td>
         <td><a href="https://drive.google.com/file/d/1PnZmCVmFwBv7ZnqMgNYj5DvmbPGM5rCx/view?usp=sharing" target="_blank" rel="noopener noreferrer">moco_r50_v2_simclr_neck-70379356.pth</a></td>
      </tr>
      <tr>
         <td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td>
         <td>selfsup/simclr/r50_bs256_ep200.py</td>
         <td>default</td>
         <td><a href="https://drive.google.com/file/d/1aZ43nSdivdNxHbM9DKVoZYVhZ8TNnmPp/view?usp=sharing" target="_blank" rel="noopener noreferrer">simclr_r50_bs256_ep200-4577e9a6.pth</a></td>
      </tr>
      <tr>
         <td></td>
         <td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td>
         <td>-&gt; MoCo v2 neck</td>
         <td><a href="https://drive.google.com/file/d/1AXpSKqgWfnj6jCgN65BXSTCKFfuIVELa/view?usp=sharing" target="_blank" rel="noopener noreferrer">simclr_r50_bs256_ep200_mocov2_neck-0d6e5ff2.pth</a></td>
      </tr>
      <tr>
         <td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td>
         <td>selfsup/byol/r50.py</td>
         <td>default</td>
         <td></td>
      </tr>
   </tbody>
</table>

## Benchmarks

### VOC07 SVM & SVM Low-shot

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th rowspan="2">Best layer</th><th rowspan="2">VOC07 SVM</th><th colspan="8">VOC07 SVM Low-shot</th></tr><tr><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>96</td></tr></thead><tbody><tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>feat5</td><td>87.17</td><td>52.99</td><td>63.55</td><td>73.7</td><td>78.79</td><td>81.76</td><td>83.75</td><td>85.18</td><td>85.97</td></tr><tr><td>Random</td><td>-</td><td>kaiming</td><td>feat2</td><td>30.22</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td></td><td></td><td>feat5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>feat4</td><td>67.38</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>feat5</td><td>74.26</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>feat5</td><td>74.50</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>selfsup/npid/r50_ensure_neg.py</td><td>ensure_neg=True</td><td>feat5</td><td>75.70</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>feat5</td><td>78.42</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>feat5</td><td>79.18</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>feat5</td><td>84.05</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>selfsup/moco/r50_v2_simclr_neck.py</td><td>-&gt; SimCLR neck<br></td><td>feat5</td><td>84.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>feat5</td><td>78.95</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td>feat5</td><td>77.65</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50.py</td><td>default</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></tbody></table>

### ImageNet Linear Classification

**Note**
* Config: `configs/benchmarks/linear_classification/imagenet/r50_multihead.py` for ImageNet (Multi) and `configs/benchmarks/linear_classification/imagenet/r50_last.py` for ImageNet (Last).
* For DeepCluster, use the corresponding one with `_sobel`.
* ImageNet (Multi) evaluates features in around 9k dimensions from different layers. Top-1 result of the last epoch is reported.
* ImageNet (Last) evaluates the last feature after global average pooling, e.g., 2048 dimensions for resnet50. The best top-1 result among all epochs is reported.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th colspan="5">ImageNet (Multi)</th><th>ImageNet (Last)</th></tr><tr><td>feat1</td><td>feat2</td><td>feat3</td><td>feat4</td><td>feat5</td><td>avgpool</td></tr></thead><tbody><tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>15.18</td><td>33.96</td><td>47.86</td><td>67.56</td><td>76.17</td><td>74.12</td></tr><tr><td>Random</td><td>-</td><td>kaiming</td><td>11.37</td><td>16.21</td><td>13.47</td><td>9.07</td><td>6.54</td><td>4.35</td></tr><tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>14.76</td><td>31.29</td><td>45.77</td><td>49.31</td><td>40.20</td><td>38.83</td></tr><tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>12.89</td><td>34.30</td><td>44.91</td><td>54.99</td><td>49.09</td><td>47.01</td></tr><tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>12.78</td><td>30.81</td><td>43.88</td><td>57.71</td><td>51.68</td><td>46.92</td></tr><tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>14.28</td><td>31.20</td><td>40.68</td><td>54.46</td><td>56.61</td><td>56.60</td></tr><tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>14.76</td><td>31.82</td><td>42.44</td><td>55.76</td><td>57.70</td><td>53.42</td></tr><tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>15.32</td><td>33.08</td><td>44.68</td><td>57.27</td><td>60.60</td><td>61.02</td></tr><tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>15.35</td><td>34.57</td><td>45.81</td><td>60.96</td><td>66.72</td><td>67.02</td></tr><tr><td></td><td>selfsup/moco/r50_v2_simclr_neck.py</td><td>-&gt; SimCLR neck<br></td><td>15.19</td><td>32.54</td><td>43.12</td><td>60.36</td><td>67.08</td><td>65.39</td></tr><tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>17.09</td><td>31.37</td><td>41.38</td><td>54.35</td><td>61.57</td><td>60.06</td></tr><tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td>16.97</td><td>31.88</td><td>41.73</td><td>54.33</td><td>59.94</td><td>58.00</td></tr><tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50.py</td><td>default</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></tbody></table>

### Places205 Linear Classification

Coming soon.

### ImageNet Semi-Supervised Classification

**Note**
* In this benchmark, the necks or heads are removed and only the backbone CNN is evaluated by appending a linear classification head. All parameters are fine-tuned.
* Config: under `configs/benchmarks/semi_classification/imagenet_1percent/` for 1% data, and `configs/benchmarks/semi_classification/imagenet_10percent/` for 10% data.
* When training with 1% ImageNet, we find hyper-parameters especially the learning rate greatly influence the performance. Hence, we prepare a list of settings with the base learning rate from \{0.001, 0.01, 0.1\} and the learning rate multiplier for the head from \{1, 10, 100\}. We choose the best performing setting for each method.
* Please use `--deterministic` in this benchmark.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th rowspan="2">Optimal setting for ImageNet 1%</th><th colspan="2">ImageNet 1%</th></tr><tr><td>top-1</td><td>top-5</td></tr></thead><tbody><tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>r50_lr0_001_head100.py</td><td>68.68</td><td>88.87</td></tr><tr><td>Random</td><td>-</td><td>kaiming</td><td>r50_lr0_01_head1.py</td><td>1.56</td><td>4.99</td></tr><tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>18.98</td><td>44.05</td></tr><tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>r50_lr0_01_head1_sobel.py</td><td>33.44</td><td>58.62</td></tr><tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>27.95</td><td>54.37</td></tr><tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>r50_lr0_1_head100.py</td><td>32.39</td><td>61.02</td></tr><tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>33.15</td><td>61.30</td></tr><tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>38.71</td><td>67.90</td></tr><tr><td></td><td>selfsup/moco/r50_v2_simclr_neck.py</td><td>-&gt; SimCLR neck<br></td><td>r50_lr0_01_head100.py</td><td>31.37<br></td><td>59.65</td></tr><tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>36.09</td><td>64.50</td></tr><tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td>r50_lr0_01_head100.py</td><td>36.31</td><td>64.68</td></tr><tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50.py</td><td>default</td><td></td><td></td><td></td></tr></tbody></table>

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th rowspan="2">Optimal setting for ImageNet 10%</th><th colspan="2">ImageNet 10%</th></tr><tr><td>top-1</td><td>top-5</td></tr></thead><tbody><tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>r50_lr0_001_head10.py</td><td>74.53</td><td>92.19</td></tr><tr><td>Random</td><td>-</td><td>kaiming</td><td>r50_lr0_01_head1.py</td><td>21.78</td><td>44.24</td></tr><tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>54.75</td><td>80.21</td></tr><tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>r50_lr0_01_head1_sobel.py</td><td>52.94</td><td>77.96</td></tr><tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>57.22</td><td>81.39</td></tr><tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>r50_lr0_1_head10.py</td><td>58.15</td><td>82.55</td></tr><tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>60.08</td><td>84.02</td></tr><tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>61.64</td><td>84.90</td></tr><tr><td></td><td>selfsup/moco/r50_v2_simclr_neck.py</td><td>-&gt; SimCLR neck<br></td><td></td><td>60.60</td><td>84.29</td></tr><tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>58.46</td><td>82.60</td></tr><tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td></td><td>58.38</td><td>82.53</td></tr><tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50.py</td><td>default</td><td></td><td></td><td></td></tr></tbody></table>

### PASCAL VOC07+12 Object Detection

Coming soon.
