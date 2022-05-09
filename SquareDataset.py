import csv
import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset


class SquareDataset(VisionDataset):

    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images_input = []
        self.images_ground = []
        self.root = root


        image_dir_input = os.path.join(self.root, 'input/')
        image_dir_ground = os.path.join(self.root, 'ground/')

        for img_file in os.listdir(image_dir_input):
            self.images_input.append(os.path.join(image_dir_input, img_file))
        for img_file in os.listdir(image_dir_ground):
            self.images_ground.append(os.path.join(image_dir_ground, img_file))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:


        """
        image_input = Image.open(self.images_input[index])
        image_input_tensor = transforms.ToTensor()(image_input)
        image_ground = Image.open(self.images_ground[index])
        image_ground_tensor = transforms.ToTensor()(image_ground)

        return image_input_tensor, image_ground_tensor



    def __len__(self) -> int:
        return len(self.images_input)