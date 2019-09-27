**Running VA_NN using the docker image**

##### Build docker

```
cd ./docker
docker build -t view_adaptive/pytorch1.2:jiaming_huang -f Dockerfile .
```

##### Run docker

We only consider the gpu case, and you have to install nvidia-docker. You can reference to [github](https://github.com/NVIDIA/nvidia-docker)

`docker run --gpus all --name va_cnn -p 6006:6006 -v .../NTU-RGB+D/nturgb+d_skeletons/:/workspace/data/NTU-RGB+D/nturgb+d_skeletons -v .../resnet50.pth:/workspace/weights/resnet50.pth -it view_adaptive/pytorch1.2:jiaming_huang`
