## Analysis of Robs data


### Instructions for use/installation:
1. Data can be downloaded from [https://gin.g-node.org/vdplasthijs/S1S2_mechanisms_data](https://gin.g-node.org/vdplasthijs/S1S2_mechanisms_data). 
2. Clone/download the two dependency repositories: [reproducible figure visualisation](https://github.com/vdplasthijs/reproducible_figures) and [Vape](https://github.com/Packer-Lab/Vape). 
3. Add your local data paths to the data folder (subfolder `S1S2_mechanisms_data/pkl_files/`) and the _rfv_ and _Vape_ repos to the `data_paths.json` file (using `username_hostname` as key). See `newuser_newcomputer` in that json file as an example template. `s2p_path` can be left blank for almost all purposes. 
4. Install environment with necessary packages:
- Install a conda environment with the necessary packages using `s1s2mech.yml`. (Using `conda env create -f s1s2mech.yml`). After conda installs:
- run `pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib`
- `pip install suite2p`. **WARNING**: This package has massive dependencies (>2GB of packages related to torch & cuda). 
5. See `jupyter/thijs/Example data loading.ipynb` for a brief tutorial on how to load the data and use the main data structures. 

Then, everything should work (as far as tested). If not - let Thijs know (via Github issue/email)!!
