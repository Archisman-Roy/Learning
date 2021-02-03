# download required data files
mkdir imagenet-data
cd imagenet-data

# Imageneatte: https://github.com/fastai/imagenette
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz # 160 PX download
tar -xvzf imagenette2-160.tgz # extract
rm imagenette2-160.tgz
mkdir out
cd out 
mkdir models
mkdir tblogs