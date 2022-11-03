# Data Flow

- [Data Flow](#data-flow)
  - [Data flow between dataloader and model](#data-flow-between-dataloader-and-model)
    - [Data from dataset](#data-from-dataset)
    - [Data from dataloader](#data-from-dataloader)
    - [Data from data preprocessor](#data-from-data-preprocessor)

Data flow defines how data should be passed between two isolated modules, e.g. dataloader and model, as shown below.

<div align="left">
<img src="https://user-images.githubusercontent.com/30762564/185855134-89f5be9e-39ca-4da4-bd87-7cf26e80ab2f.png" width="70%"/>
</div>

In MMSelfSup, we mainly focus on the data flow between dataloader and model, and between model and visualizer. As for the
data flow between model and metric, please refer to the docs in other repos, e.g. [MMClassification](https://github.com/open-mmlab/mmclassification).
Also for data flow between model and visualizer, you can refer to [visualization](../user_guides/visualization.md)

## Data flow between dataloader and model

The data flow between dataloader and model can be generally split into three parts, i) use `PackSelfSupInputs` to pack
data from previous transformations into a dictionary, ii) use `collate_fn` to stack a list of tensors into a batched tensor,
iii) data preprocessor will move all these data to target device, e.g. GPUS, and unzip the dictionary from the dataloader
into a tuple, containing the input images and meta info (`SelfSupDataSample`).

### Data from dataset

In MMSelfSup, before feeding into the model, data should go through a series of transformations, called `pipeline`, e.g. `RandomResizedCrop` and `ColorJitter`. No matter how many transformations in the pipeline, the last transformation is `PackSelfSupInputs`. `PackSelfSupInputs` will
pack these data from previous transformations into a dictionary. The dictionary contains two parts, namely, `inputs` and `data_samples`.

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
        packed_results['data_samples'] = data_sample

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
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:

        assert isinstance(data,
                          dict), 'Please use default_collate in dataloader, \
            instead of pseudo_collate.'

        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)
        # channel transform
        if self._channel_conversion:
            batch_inputs = [
                _input[:, [2, 1, 0], ...] for _input in batch_inputs
            ]

        # Convert to float after channel conversion to ensure
        # efficiency
        batch_inputs = [input_.float() for input_ in batch_inputs]

        # Normalization. Here is what is different from
        # :class:`mmengine.ImgDataPreprocessor`. Since there are multiple views
        # for an image for some  algorithms, e.g. SimCLR, each item in inputs
        # is a list, containing multi-views for an image.
        if self._enable_normalize:
            batch_inputs = [(_input - self.mean) / self.std
                            for _input in batch_inputs]

        return batch_inputs, batch_data_samples
```
