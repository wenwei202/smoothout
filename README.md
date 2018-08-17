# SmoothOut: Smoothing Out Sharp Minima for Generalization in Large-Batch Deep Learning

## Related paper
This is the code of paper  https://arxiv.org/abs/1805.07898 . In large-batch training of deep neural networks, accuracy loss is observed. [Some](https://arxiv.org/abs/1609.04836) hypothesize that this is because of the existance of sharp minima in deep neural networks. 
In the paper, smoothout method is proposed to eliminate sharp minima in deep neural networks in large-batch training, such that we can use a larger batch size to improve the scalability of distributed training of deep neural networks.

This code is based on [bigBatch](https://github.com/eladhoffer/bigBatch).

Tutorial to be updated soon.

## Usage

```
python evaluate.py --model alexnet --dataset imagenet -b 100 --gpus 1 --evaluate TrainingResults/2018-03-17_11-28-33/model_best.pth.tar --ss 0.006 --mode train --no-augment
```

```
python visualize_sharpness.py --model resnet --dataset cifar10 -b 100 --gpus 1 --evaluate TrainingResults/2018-02-20_19-47-52/model_best.pth.tar  --slave-evaluate TrainingResults/2018-02-20_19-47-18/model_best.pth.tar  --no-visualize_train
```

```
python visualize2_sharpness.py --model resnet --dataset cifar10 -b 100 --gpus 1 --evaluate TrainingResults/2018-02-20_19-47-18/model_best.pth.tar  --no-visualize_train --alpha="-0.3:0.02:0.31"
```

```
python measure_sharpness.py --model cifar100_shallow --dataset cifar100 --b 100 --gpus 1 --evaluate TrainingResults/2018-02-20_14-05-28/model_best.pth.tar
```

## Dependencies

- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization `conda install bokeh=0.12.0`


## Data
- Configure your dataset path at **data.py**.
- To get the ILSVRC data, you should register on their site for access: <http://www.image-net.org/> and
```
mkdir -p pytorch-gen/imagenet
cd pytorch-gen/imagenet
ln -s ${YOUR_IMAGENET_PATH}/train/ train
ln -s ${YOUR_IMAGENET_PATH}/validation/ val
```


## Model configuration

Network model is defined by writing a <modelname>.py file in <code>models</code> folder, and selecting it using the <code>model</code> flag. Model function must be registered in <code>models/\_\_init\_\_.py</code>
The model function must return a trainable network. It can also specify additional training options such optimization regime (either a dictionary or a function), and input transform modifications.

e.g for a model definition:

```python
class Model(nn.Module):

    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.model = nn.Sequential(...)

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            15: {'lr': 1e-3, 'weight_decay': 0}
        }

        self.input_transform = {
            'train': transforms.Compose([...]),
            'eval': transforms.Compose([...])
        }
    def forward(self, inputs):
        return self.model(inputs)

 def model(**kwargs):
        return Model()
```
