# Data Flow

- [Data Flow](#data-flow)
  - [Data flow between dataloader and model](#data-flow-between-dataloader-and-model)
    - [Data from dataset](#data-from-dataset)
    - [Data from dataloader](#data-from-dataloader)
    - [Data from data preprocessor](#data-from-data-preprocessor)

Data flow defines how data should be passed between two isolated modules, e.g. dataloader and model, as shown below.

<div align="left">
<img src="https://user-images.githubusercontent.com/30762564/185855134-89f5be9e-39ca-4da4-bd87-7cf26e80ab2f.png" width="40%"/>
</div>

In MMSelfSup, we mainly focus on the data flow between dataloader and model, and between model and visualizer. As for the
data flow between model and metric, please refer to the docs in other repos, e.g. [MMClassification](https://github.com/open-mmlab/mmclassification).
Also for data flow between model and visualizer, you can refer to [visualization](../user/guides/visualization.md)

## Data flow between dataloader and model

The data flow between dataloader and model can be generally split into three parts, i) use `PackSelfSupInputs` to pack
data from previous transformations into a dictionary, ii) use `collect_fn` to stack a list of tensors into a batched tensor,
iii) data preprocessor will move all these data to target device, e.g. GPUS, and unzip the dictionary from the dataloader
into a tuple, containing the input images and meta info (`SelfSupDataSample`).

### Data from dataset

In MMSelfSup, before feeding into the model, data should go through a series of transformations, called `pipeline`, e.g. `RandomResizedCrop` and `ColorJitter`. No matter how many transformations in the pipeline, the last transformation is `PackSelfSupInputs`. `PackSelfSupInputs` will
pack these data from previous transformations into a dictionary. The dictionary contains two parts, namely, `inputs` and `data_sample`.

```python

# We omit some unimportant code here

class PackSelfSupInputs(BaseTransform):

    def transform(self,
                  results: Dict) -> Dict[torch.Tensor, SelfSupDataSample]:

        packed_results = dict()
        if self.key in results:
            ...
            packed_results['inputs'] = img

        ...
        packed_results['data_sample'] = data_sample

        return packed_results
```

Note: `inputs` contains a list of images, e.g. the multi-views in contrastive learning. Even a single view,
`PackSelfSupInputs` will still put it into a list.

### Data from dataloader

After receiving a list of dictionary from dataset, `collect_fn` in dataloader will gather `inputs` in each dict
and stack them into a batched tensor. In addition, `data_sample` in each dict will be also collected in a list.
Then, it will output a dict, containing the same keys with those of the dict in the received list. Finally, dataloader
will output the dict from the `collect_fn`.

### Data from data preprocessor

Data preprocessor is the last step to process the data before feeding into the model. It will apply image normalization, convert BGR to RGB
and move all data to the target device, e.g. GPUs. After above steps, it will output a tuple, containing a list of batched images, and a list
of data samples.

```python
class SelfSupDataPreprocessor(ImgDataPreprocessor):

    def forward(
            self,
            data: Sequence[dict],
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:

        inputs, batch_data_samples = self.collate_data(data)
        # channel transform
        if self.channel_conversion:
            inputs = [[img_[[2, 1, 0], ...] for img_ in _input]
                      for _input in inputs]

        # Normalization. Here is what is different from
        # :class:`mmengine.ImgDataPreprocessor`. Since there are multiple views
        # for an image for some  algorithms, e.g. SimCLR, each item in inputs
        # is a list, containing multi-views for an image.
        if self._enable_normalize:
            inputs = [[(img_ - self.mean) / self.std for img_ in _input]
                      for _input in inputs]

        batch_inputs = []
        for i in range(len(inputs[0])):
            cur_batch = [img[i] for img in inputs]
            batch_inputs.append(torch.stack(cur_batch))

        return batch_inputs, batch_data_samples
```
