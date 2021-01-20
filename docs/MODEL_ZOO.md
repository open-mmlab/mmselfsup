# Model Zoo

**OpenSelfSup needs your contribution!
Since we don't have sufficient GPUs to run these large-scale experiments, your contributions, including parameter studies, reproducing of results, implementing new methods, etc, are essential to make OpenSelfSup better. Your contribution will be recorded in the below table, top contributors will be included in the author list of OpenSelfSup!**

## Pre-trained model download links and speed test.
**Note**
* If not specifically indicated, the testing GPUs are NVIDIA Tesla V100.
* The table records the implementors who implemented the methods (either by themselves or refactoring from other repos), and the experimenters who performed experiments and reproduced the results. The experimenters should be responsible for the evaluation results on all the benchmarks, and the implementors should be responsible for the implementation as well as the results; If the experimenter is not indicated, an implementator is the experimenter by default.

<table><thead><tr><th>Method (Implementator)</th><th>Config (Experimenter)</th><th>Remarks</th><th>Download link</th><th>Batch size</th><th>Epochs</th></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py">ImageNet</a></td><td>-</td><td>torchvision</td><td><a href="https://drive.google.com/file/d/11xA3TOcbD0qOrwpBfYonEDeseE1wMfBh/view?usp=sharing">imagenet_r50-21352794.pth</a></td><td>-</td><td>-</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td><a href="https://drive.google.com/file/d/1UaFTjd6sbKkZEE-f58Zv30bnx7C1qJBb/view?usp=sharing">random_r50-5d0fa71b.pth</a></td><td>-</td><td>-</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf">Relative-Loc</a> (<a href="https://github.com/Jiahao000">@Jiahao000</a>)</td><td>selfsup/relative_loc/r50.py</td><td>default</td><td><a href="https://drive.google.com/file/d/1ibk1BI3PFQxZqcxuDfHs3n7JnWKCgl8x/view?usp=sharing">relative_loc_r50-342c9097.pth</a></td><td>512</td><td>70</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728">Rotation-Pred</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td><a href="https://drive.google.com/file/d/1t3oClmIvQ0p8RZ0V5yvQFltzjqBO823Y/view?usp=sharing">rotation_r50-cfab8ebb.pth</a></td><td>128</td><td>70</td></tr>
<tr><td><a href="https://arxiv.org/abs/1807.05520">DeepCluster</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/deepcluster/r50.py</td><td>default</td><td><a href="https://drive.google.com/file/d/1GxgP7pI18JtFxDIC0hnHOanvUYajoLlg/view?usp=sharing">deepcluster_r50-bb8681e2.pth</a></td><td>512</td><td>200</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978">NPID</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/npid/r50.py</td><td>default</td><td><a href="https://drive.google.com/file/d/1sm6I3Y5XnCWdbmeLSF4YupUtPe5nRQMI/view?usp=sharing">npid_r50-dec3df0c.pth</a></td><td>256</td><td>200</td></tr>
<tr><td></td><td>selfsup/npid/r50_ensure_neg.py</td><td>ensure_neg=True</td><td><a href="https://drive.google.com/file/d/1FldDrb6kzF3CZ7737mwCXVI6HE2aCSaF/view?usp=sharing">npid_r50_ensure_neg-ce09b7ae.pth</a></td><td></td><td></td></tr>
<tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf">ODC</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/odc/r50_v1.py (<a href="https://github.com/Jiahao000">@Jiahao000</a>)</td><td>default</td><td><a href="https://drive.google.com/file/d/1EdhJeZAyMsD_pEW7uMhLzos5xZLdariN/view?usp=sharing">odc_r50_v1-5af5dd0c.pth</a></td><td>512</td><td>440</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722">MoCo</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/moco/r50_v1.py</td><td>default</td><td><a href="https://drive.google.com/file/d/1ANXfnoT8yBQQBBqR_kQLQorK20l65KMy/view?usp=sharing">moco_r50_v1-4ad89b5c.pth</a></td><td>256</td><td>200</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/moco/r50_v2.py</td><td>default</td><td><a href="https://drive.google.com/file/d/1ImO8A3uWbrTx21D1IqBDMUQvpN6wmv0d/view?usp=sharing">moco_r50_v2-e3b0c442.pth</a></td><td>256</td><td>200</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709">SimCLR</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td><a href="https://drive.google.com/file/d/1aZ43nSdivdNxHbM9DKVoZYVhZ8TNnmPp/view?usp=sharing">simclr_r50_bs256_ep200-4577e9a6.pth</a></td><td>256</td><td>200</td></tr>
<tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td><a href="https://drive.google.com/file/d/1AXpSKqgWfnj6jCgN65BXSTCKFfuIVELa/view?usp=sharing">simclr_r50_bs256_ep200_mocov2_neck-0d6e5ff2.pth</a></td><td></td><td></td></tr>
<!--
<tr><td><a href="https://arxiv.org/abs/2006.07733">BYOL</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/byol/r50_bs4096_ep200.py (<a href="https://github.com/xieenze">@xieenze</a>)</td><td>default</td><td><a href="https://drive.google.com/file/d/1Whj3j5E3ShQj_VufjrJSzWiq1xcZZCXN/view?usp=sharing">byol_r50-e3b0c442.pth</a></td><td>4096</td><td>200</td></tr>
-->
<tr><td><a href="https://arxiv.org/abs/2006.07733">BYOL</a> (<a href="https://github.com/XiaohangZhan">@XiaohangZhan</a>)</td><td>selfsup/byol/r50_bs256_accumulate16_ep300.py (<a href="https://github.com/scnuhealthy">@scnuhealthy</a>)</td><td>default</td><td><a href="https://drive.google.com/file/d/12Zu9r3fE8qKF4OW6WQXa5Ec6VuA2m3j7/view?usp=sharing">byol_r50_bs256_accmulate16_ep300-5df46722.pth</a></td><td>256</td><td>300</td></tr>
<tr><td></td><td>selfsup/byol/r50_bs2048_accumulate2_ep200_fp16.py (<a href="https://github.com/xieenze">@xieenze</a>)</td><td>default</td><td><a href="https://drive.google.com/file/d/1Poj-rxIebE1ykJhB8EeliSxWiCGP_rbD/view?usp=sharing">byol_r50_bs2048_accmulate2_ep200-e3b0c442.pth</a></td><td>2048</td><td>200</td></tr>
</tbody></table>


## Benchmarks

### VOC07 SVM & SVM Low-shot

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th rowspan="2">Best layer</th><th rowspan="2">VOC07 SVM</th><th colspan="8">VOC07 SVM Low-shot</th></tr>
<tr><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>96</td></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>feat5</td><td>87.17</td><td>52.99</td><td>63.55</td><td>73.7</td><td>78.79</td><td>81.76</td><td>83.75</td><td>85.18</td><td>85.97</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td>feat2</td><td>30.54</td><td>9.15</td><td>9.39</td><td>11.09</td><td>12.3</td><td>14.3</td><td>17.41</td><td>21.32</td><td>23.77</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>feat4</td><td>64.78</td><td>18.17</td><td>22.08</td><td>29.37</td><td>35.58</td><td>41.8</td><td>48.73</td><td>55.55</td><td>58.33</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>feat4</td><td>67.38</td><td>18.91</td><td>23.33</td><td>30.57</td><td>38.22</td><td>45.83</td><td>52.23</td><td>58.08</td><td>61.11</td></tr>
<tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>feat5</td><td>74.26</td><td>29.73</td><td>37.66</td><td>45.85</td><td>55.57</td><td>62.48</td><td>66.15</td><td>70.0</td><td>71.37</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>feat5</td><td>74.50</td><td>24.19</td><td>31.24</td><td>39.69</td><td>50.99</td><td>59.03</td><td>64.4</td><td>68.69</td><td>70.84</td></tr>
<tr><td></td><td>selfsup/npid/r50_ensure_neg.py</td><td>ensure_neg=True</td><td>feat5</td><td>75.70</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>feat5</td><td>78.42</td><td>32.42</td><td>40.27</td><td>49.95</td><td>59.96</td><td>65.71</td><td>69.99</td><td>73.64</td><td>75.13</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>feat5</td><td>79.18</td><td>30.03</td><td>37.73</td><td>47.64</td><td>58.78</td><td>66.0</td><td>70.6</td><td>74.6</td><td>76.07</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>feat5</td><td>84.26</td><td>43.0</td><td>52.48</td><td>63.43</td><td>71.74</td><td>76.35</td><td>78.9</td><td>81.31</td><td>82.45</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>feat5</td><td>78.95</td><td>32.45</td><td>40.76</td><td>50.4</td><td>59.01</td><td>65.45</td><td>70.13</td><td>73.58</td><td>75.35</td></tr>
<tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td>feat5</td><td>77.65</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50_bs256_accumulate16_ep300.py</td><td>default</td><td>feat5</td><td>86.58</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td>selfsup/byol/r50_bs2048_accumulate2_ep200_fp16.py</td><td>default</td><td>feat5</td><td>85.86</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>

</tbody></table>


### ImageNet Linear Classification

**Note**
* Config: `configs/benchmarks/linear_classification/imagenet/r50_multihead.py` for ImageNet (Multi) and `configs/benchmarks/linear_classification/imagenet/r50_last.py` for ImageNet (Last).
* For DeepCluster, use the corresponding one with `_sobel`.
* ImageNet (Multi) evaluates features in around 9k dimensions from different layers. Top-1 result of the last epoch is reported.
* ImageNet (Last) evaluates the last feature after global average pooling, e.g., 2048 dimensions for resnet50. The best top-1 result among all epochs is reported.
* Usually, we report the best result from ImageNet (Multi) and ImageNet (Last) to ensure fairness, since different methods achieve their best performance on different layers.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th colspan="5">ImageNet (Multi)</th><th>ImageNet (Last)</th></tr>
<tr><td>feat1</td><td>feat2</td><td>feat3</td><td>feat4</td><td>feat5</td><td>avgpool</td></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>15.18</td><td>33.96</td><td>47.86</td><td>67.56</td><td>76.17</td><td>74.12</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td>11.37</td><td>16.21</td><td>13.47</td><td>9.07</td><td>6.54</td><td>4.35</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>14.76</td><td>31.29</td><td>45.77</td><td>49.31</td><td>40.20</td><td>38.83</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>12.89</td><td>34.30</td><td>44.91</td><td>54.99</td><td>49.09</td><td>47.01</td></tr>
<tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>12.78</td><td>30.81</td><td>43.88</td><td>57.71</td><td>51.68</td><td>46.92</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>14.28</td><td>31.20</td><td>40.68</td><td>54.46</td><td>56.61</td><td>56.60</td></tr>
<tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>14.76</td><td>31.82</td><td>42.44</td><td>55.76</td><td>57.70</td><td>53.42</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>15.32</td><td>33.08</td><td>44.68</td><td>57.27</td><td>60.60</td><td>61.02</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>14.74</td><td>32.81</td><td>44.95</td><td>61.61</td><td>66.73</td><td>67.69</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>17.09</td><td>31.37</td><td>41.38</td><td>54.35</td><td>61.57</td><td>60.06</td></tr>
<tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td>16.97</td><td>31.88</td><td>41.73</td><td>54.33</td><td>59.94</td><td>58.00</td></tr>
<!--
<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50_bs4096_ep200.py</td><td>default</td><td>16.70</td><td>34.22</td><td>46.61</td><td>60.78</td><td>69.14</td><td>67.10</td></tr>
-->
<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50_bs256_accumulate16_ep300.py</td><td>default</td><td>14.07</td><td>34.44</td><td>47.22</td><td>63.08</td><td>72.35</td><td></td></tr>
<tr><td></td><td>selfsup/byol/r50_bs2048_accumulate2_ep200_fp16.py</td><td>default</td><td>15.52</td><td>34.50</td><td>47.22</td><td>62.78</td><td>71.61</td><td></td></tr>
</tbody></table>

### Places205 Linear Classification

**Note**
* Config: `configs/benchmarks/linear_classification/places205/r50_multihead.py`.
* For DeepCluster, use the corresponding one with `_sobel`.
* Places205 evaluates features in around 9k dimensions from different layers. Top-1 result of the last epoch is reported.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th colspan="5">Places205</th></tr>
<tr><td>feat1</td><td>feat2</td><td>feat3</td><td>feat4</td><td>feat5</td></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>21.27</td><td>36.10</td><td>43.03</td><td>51.38</td><td>53.05</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td>17.19</td><td>21.70</td><td>19.23</td><td>14.59</td><td>11.73</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>21.07</td><td>34.86</td><td>42.84</td><td>45.71</td><td>41.45</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>18.65</td><td>35.71</td><td>42.28</td><td>45.98</td><td>43.72</td></tr>
<tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>18.80</td><td>33.93</td><td>41.44</td><td>47.22</td><td>42.61</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>20.53</td><td>34.03</td><td>40.48</td><td>47.13</td><td>47.73</td></tr>
<tr><td><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>20.94</td><td>34.78</td><td>41.19</td><td>47.45</td><td>49.18</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>21.13</td><td>35.19</td><td>42.40</td><td>48.78</td><td>50.70</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>21.88</td><td>35.75</td><td>43.65</td><td>49.99</td><td>52.57</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>22.55</td><td>34.14</td><td>40.35</td><td>47.15</td><td>51.64</td></tr>
<tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody></table>

### ImageNet Semi-Supervised Classification

**Note**
* In this benchmark, the necks or heads are removed and only the backbone CNN is evaluated by appending a linear classification head. All parameters are fine-tuned.
* Config: under `configs/benchmarks/semi_classification/imagenet_1percent/` for 1% data, and `configs/benchmarks/semi_classification/imagenet_10percent/` for 10% data.
* When training with 1% ImageNet, we find hyper-parameters especially the learning rate greatly influence the performance. Hence, we prepare a list of settings with the base learning rate from \{0.001, 0.01, 0.1\} and the learning rate multiplier for the head from \{1, 10, 100\}. We choose the best performing setting for each method.
* Please use `--deterministic` in this benchmark.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th rowspan="2">Optimal setting for ImageNet 1%</th><th colspan="2">ImageNet 1%</th></tr>
<tr><td>top-1</td><td>top-5</td></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>r50_lr0_001_head100.py</td><td>68.68</td><td>88.87</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td>r50_lr0_01_head1.py</td><td>1.56</td><td>4.99</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>16.48</td><td>40.37</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>18.98</td><td>44.05</td></tr>
<tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>r50_lr0_01_head1_sobel.py</td><td>33.44</td><td>58.62</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>27.95</td><td>54.37</td></tr>
<tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>r50_lr0_1_head100.py</td><td>32.39</td><td>61.02</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>33.15</td><td>61.30</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>39.07</td><td>68.31</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>36.09</td><td>64.50</td></tr>
<tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td>r50_lr0_01_head100.py</td><td>36.31</td><td>64.68</td></tr>
</tbody></table>

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th rowspan="2">Optimal setting for ImageNet 10%</th><th colspan="2">ImageNet 10%</th></tr>
<tr><td>top-1</td><td>top-5</td></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>r50_lr0_001_head10.py</td><td>74.53</td><td>92.19</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td>r50_lr0_01_head1.py</td><td>21.78</td><td>44.24</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>53.86</td><td>79.62</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>54.75</td><td>80.21</td></tr>
<tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>selfsup/deepcluster/r50.py</td><td>default</td><td>r50_lr0_01_head1_sobel.py</td><td>52.94</td><td>77.96</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>57.22</td><td>81.39</td></tr>
<tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>selfsup/odc/r50_v1.py</td><td>default</td><td>r50_lr0_1_head10.py</td><td>58.15</td><td>82.55</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>60.08</td><td>84.02</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>61.80</td><td>85.11</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>r50_lr0_01_head100.py</td><td>58.46</td><td>82.60</td></tr>
<tr><td></td><td>selfsup/simclr/r50_bs256_ep200_mocov2_neck.py</td><td>-&gt; MoCo v2 neck</td><td>r50_lr0_01_head100.py</td><td>58.38</td><td>82.53</td></tr>
</tbody></table>

### PASCAL VOC07+12 Object Detection

**Note**
* This benchmark follows the evluation protocols set up by MoCo.
* Config: `benchmarks/detection/configs/pascal_voc_R_50_C4_24k_moco.yaml`.
* Please follow [here](GETTING_STARTED.md#voc0712--coco17-object-detection) to run the evaluation.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th colspan="3">VOC07+12</th></tr>
<tr><td>AP50</td><td>AP</td><td>AP75</td></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>81.58</td><td>54.19</td><td>59.80</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td>59.02</td><td>32.83</td><td>31.60</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>80.36</td><td>55.13</td><td>61.18</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>80.91</td><td>55.52</td><td>61.39</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>80.03</td><td>54.11</td><td>59.50</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>81.38</td><td>55.95</td><td>62.23</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>82.24</td><td>56.97</td><td>63.43</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>79.41</td><td>51.54</td><td>55.63</td></tr>
<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL</a></td><td>selfsup/byol/r50_bs2048_accumulate2_ep200_fp16.py</td><td>default</td><td>79.60</td><td>49.00</td><td>52.80</td></tr>
</tbody></table>

### COCO2017 Object Detection

**Note**
* This benchmark follows the evluation protocols set up by MoCo.
* Config: `benchmarks/detection/configs/coco_R_50_C4_2x_moco.yaml`.
* Please follow [here](GETTING_STARTED.md#voc0712--coco17-object-detection) to run the evaluation.

<table><thead><tr><th rowspan="2">Method</th><th rowspan="2">Config</th><th rowspan="2">Remarks</th><th colspan="6">COCO2017</th></tr>
<tr><td>AP50(Box)</td><td>AP(Box)</td><td>AP75(Box)</td><td>AP50(Mask)</td><td>AP(Mask)</td><td>AP75(Mask)</td></tr></thead><tbody>
<tr><td><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py" target="_blank" rel="noopener noreferrer">ImageNet</a></td><td>-</td><td>torchvision</td><td>59.9</td><td>40.0</td><td>43.1</td><td>56.5</td><td>34.7</td><td>36.9</td></tr>
<tr><td>Random</td><td>-</td><td>kaiming</td><td>54.6</td><td>35.6</td><td>38.2</td><td>51.5</td><td>31.4</td><td>33.5</td></tr>
<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>selfsup/relative_loc/r50.py</td><td>default</td><td>59.6</td><td>40.0</td><td>43.5</td><td>56.5</td><td>35.0</td><td>37.3</td></tr>
<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>selfsup/rotation_pred/r50.py</td><td>default</td><td>59.3</td><td>40.0</td><td>43.6</td><td>56.0</td><td>34.9</td><td>37.4</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>selfsup/npid/r50.py</td><td>default</td><td>59.0</td><td>39.4</td><td>42.8</td><td>55.9</td><td>34.5</td><td>36.6</td></tr>
<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>selfsup/moco/r50_v1.py</td><td>default</td><td>60.5</td><td>40.9</td><td>44.2</td><td>57.1</td><td>35.5</td><td>37.7</td></tr>
<tr><td><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td><td>selfsup/moco/r50_v2.py</td><td>default</td><td>60.6</td><td>41.0</td><td>44.5</td><td>57.2</td><td>35.6</td><td>38.0</td></tr>
<tr><td><a href="https://arxiv.org/abs/2002.05709">SimCLR</a></td><td>selfsup/simclr/r50_bs256_ep200.py</td><td>default</td><td>59.1</td><td>39.6</td><td>42.9</td><td>55.9</td><td>34.6</td><td>37.1</td></tr>
<tr><td><a href="https://arxiv.org/abs/2006.07733">BYOL</a></td><td>selfsup/byol/r50_bs2048_accumulate2_ep200_fp16.py</td><td>default</td><td>60.6</td><td>40.2</td><td>43.3</td><td>57.0</td><td>34.9</td><td>36.7</td></tr>
</tbody></table>

