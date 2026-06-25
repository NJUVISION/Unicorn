## Example Test Commands

Lossy static attribute:

```bash
python lossy_attribute/test.py \
  --scale=5 --Vmode=1 \
  --testdata=/path/to/ply_directory \
  --piecewise_variable_bitrates=object \
  --prefix=attribute_test/8ivfb
```

Lossless static attribute:

```bash
python lossless_attribute/test.py \
  --split_channel=1 --scale=20 --block_layers=3 \
  --color_format=ycocg --normalize=0 \
  --ckptdir=/path/to/epoch_last.pth \
  --testdata=/path/to/ply_directory \
  --prefix=lossless_attribute_test
```

Dynamic lossy attribute:

```bash
python dynamic_attribute/test.py \
  --scale=5 --Vmode=1 --inter_mode=1 \
  --testdata=/path/to/dynamic_sequence_directory \
  --testdata_num=100 \
  --prefix=dynamic_attribute_test
```
