# OneHotConv

[![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-green.svg)](https://github.com/YassineYousfi/alaska/issues)
[![Generic badge](https://img.shields.io/badge/Status-Beta-ffa500.svg)](https://github.com/YassineYousfi/alaska/pulse)

This is an implementation of the OneHot CNN for JPEG steganalysis proposed in this [paper](http://www.ws.binghamton.edu/fridrich/Research/OneHot_Revised.pdf).

## Data 
Dataset preparation is not part of this script. Make sure your data follows the following structure:

```
DATA-PATH
└───QF100
    └───COVER
    │      └───TRN
    │      └───VAL
    │      └───TST
    │
    └───STEGO_PAYLOAD
           └───TRN
           └───VAL
           └───TST
```

## How to use
```
python3 train_lit_model.py --version {experiment name} --gpus {num gpus} --data-path {data path root} --stego-scheme {stego scheme name} --payload {payload}
```

## WIP
- Fix training with AMP fp16
- Enable different DCT domain and Spatial domain backbones
- Update to pytorch lightning 1.0

## Dependecies

Python 3.5+, pytorch 1.4+ and dependencies listed in `requirements.txt`.

## References

Please consider citing our paper if you find this repository useful.

```
@article{9091221,
  author={Y. {Yousfi} and J. {Fridrich}},
  journal={IEEE Signal Processing Letters}, 
  title={An Intriguing Struggle of CNNs in JPEG Steganalysis and the OneHot Solution}, 
  year={2020},
  volume={27},
  number={},
  pages={830-834},
  doi={10.1109/LSP.2020.2993959}}
```