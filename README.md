# "What happens if..." Learning to Predict the Effect of Forces in Images
This is the source code for a deep net that predicts the effect of applying a force to an object shown in a static image.

### Citation
If you find the code useful in your research, please consider citing:
```
@inproceedings{mottaghiECCV16,
    Author = {Roozbeh Mottaghi and Mohammad Rastegari and Abhinav Gupta and Ali Farhadi},
    Title = {``What happens if..." Learning to Predict the Effect of Forces in Images},
    Booktitle = {ECCV},
    Year = {2016}
}
```

### Requirements
This code is written in Lua, based on [Torch](http://torch.ch). If you use [Ubuntu 14.04+](http://ubuntu.com), you can follow [these instructions](https://github.com/facebook/fbcunn/blob/master/INSTALL.md) to install torch.

You need to download the [ForScene dataset](https://s3-us-west-2.amazonaws.com/ai2-vision-datasets/ForScene_dataset/ForScene.tar.gz) (2GB). Extract the files and set the correct paths in `setting_options.lua`.

### Training
To train the model, run:
```
th main.lua train
```

### Test
Set the path to the learned model in `setting_options.lua` (`config.initModelPath.fullNN`). You also need to set the number of batches `config.nIter` and the batch size `config.batchSize`. To run the test:
```
th main.lua test
```
Our released files contain a pre-trained model `Model_iter_15000.t7`, which is a model trained using AlexNet and object masks (no depth). You can set the path to this file and run a test to make sure you can re-produce the result (16.5% accuracy) in the paper.

### Simulations
We have also provided the code for generating the simulations. You need to load `scene_gen.blend` in Blender game engine.

### License
This code is released under MIT License.