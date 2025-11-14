# Prepare Datasets for EVOVLE

# Tips
- Keep in mind that it does not perform pre-trained (COCO, Brust...) unlike traditional VOS methods.

## Generate EventVoxel for dataset:

- To generate EventVoxel for datasets, convert eventstream to eventvoxel type from `generate_voxel.py`

EVOVLE has builtin support for two datasets.

## Expected dataset structure for [LLE-VOS]:

```
LLE_VOS/
  Lowlight_Images/
  EventVoxel/
  Annotations/
  Lowlight_event/
  Normallight_Images/
  {train, val_indoor, val_outdoor, val}.txt
```

## Expected dataset structure for [LLE-DAVIS]:

```
LLE_DAVIS/
  trainval_480p/
     Annotations/
     EventVoxel/
     JPEGImages/
     Lowlight_Images/
     Lowlight_event/
     ImageSets/
```

