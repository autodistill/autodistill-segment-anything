<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Segment Anything Module

This repository contains the code supporting the Segment Anything base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) is a segmentation model developed by Meta AI Research. Segment Anything can segment an entire image into masks, or use points to segment specific parts of an object. You can use Segment Anything with Autodistill to segment objects. Segment Anything does not assign classes, so you should use the model with a tool like Grounding DINO or GPT-4V.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SAM Autodistill documentation](https://autodistill.github.io/autodistill/base_models/SAM/).

## Installation

To use SAM with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-segment-anything
```

## Quickstart

```python
from autodistill_segment_anything import SegmentAnything

base_model = SegmentAnything(None)

masks = base_model.predict("./image.jpeg")

print(masks)
```


## License

This project is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!