
## Goal
In this repository, we implement a fully convolutional regression  networks (FCRNs) approach for regression of a density map in order to get density map of cell pictures for biomedical goals.
## Generation of Database 

Generating squares randomly positionned in a 100x100 image.
First step is to randomly position the squares, then get the coordinates of the center of squares, and apply Gaussian filter over them for ground truth.
Example from dataset : 

<p align="center">
  <img alt="Light" src="./Images/image_in_7_22.png" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./Images/image_gth_4_17.png" width="30%">
</p>

## Network description 

<p align="center">
  <img alt="Light" src="./Images/FCRN.png" width="50%">
 </p>

## Test on example : 

<p align="center">
  <img alt="Light" src="./Images/image_4_35.png" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./Images/4_35(1).png" width="30%">
</p>
<p align="center">
  <img alt="Light" src="./Images/image_8_9.png" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./Images/8_9(1).png" width="30%">
</p>

## Requirements : 

```bash
git clone GIT_repo
pip install -r requirements.txt
```

## Training/Validation 
  # SquareDataset

The `SquareDataset` is a custom dataset class designed for use with PyTorch. It extends the `VisionDataset` class and is specifically tailored for tasks involving paired image data, such as image-to-image translation.

## Class Overview

### Constructor Parameters

- `root` (str): The root directory of the dataset, containing 'input' and 'ground' subdirectories for input and ground truth images respectively.

- `transform` (Optional[Callable]): A function/transform to apply to input and ground truth images.

- `target_transform` (Optional[Callable]): A function/transform to apply to target images.

- `transforms` (Optional[Callable]): A function/transform to apply to both input and target images.

### Methods

- `__getitem__(index: int) -> Tuple[Any, Any]`: Retrieves an item at a given index, returning a tuple with the input and ground truth images.

- `__len__() -> int`: Returns the total number of items in the dataset.

## Architecture

The dataset class leverages the PyTorch library, utilizing the `PIL` (Pillow) for image handling and `torchvision.transforms` for image transformations. It efficiently organizes and loads paired images from specified directories, making them easily accessible for training machine learning models.

## Example Usage

```python
from square_dataset import SquareDataset
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Initialize the dataset with transformations
dataset = SquareDataset(root='path_to_dataset_root', transform=transform)

# Access the first item
input_image, ground_truth_image = dataset[0]

## Testing : 
  Building
## 🔗 Links
- [ Paper : Microscopy cell counting and detection with fully convolutional regression networks)](https://www.tandfonline.com/doi/abs/10.1080/21681163.2016.1149104?journalCode=tciv20 "Microscopy cell counting and detection with fully convolutional regression networks")
## Authors
```
  author    = {BOUKHEMS Anass and
               Taha Mohammed Elqandili},
  title     = {Cellular counting using convolutional network},
  year      = {2021},
  emails    =  {boukhemsanass0@gmail.com,
              elqandili.taha@gmail.com }

```
