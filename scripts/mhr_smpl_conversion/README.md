## Convert MHR results from SAM-Body4D to SMPL
#### 1. Install extra dependencies
```
conda create -n mhr2smpl python=3.12 -y
conda activate mhr2smpl
# pip install pymomentum-cpu fails on some envs
conda install -c conda-forge pymomentum-cpu -y
pip install scikit-learn smplx mhr tqdm opencv-python einops colorlog
conda install pytorch3d -c pytorch3d-nightly --no-deps -y
pip install chumpy --no-build-isolation
```

More info about [PyMomentum](https://facebookresearch.github.io/momentum/pymomentum/user_guide/getting_started).

#### 2. Call 


## SMPL Model Preparation

#### 1. Download [SMPL model](https://smpl.is.tue.mpg.de/index.html) (python version, registration required)

- Consider the neutral version (basicmodel_f_lbs_10_207_0_v1.1.0.pkl)

#### 2. Remove Chumpy from SMPL model

- Clone this repo: [https://github.com/vchoutas/smplx](https://github.com/vchoutas/smplx)

- Create an env with python=2.7
```
conda create -n py27 python=2.7 -y
conda activate py27
pip install tqdm numpy scipy chumpy
```

- Run: 
```
python tools/clean_ch.py --input-models body_models/smpl/SMPL_NEUTRAL.pkl --output-folder output-fol
```

#### 3. Use the resulting pkl in conversion scripts
