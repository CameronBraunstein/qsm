# Quantum-Hybrid Stereo Matching with Nonlinear Regularization and Spatial Pyramids
[Cameron Braunstein](https://cameronbraunstein.github.io/), [Vladislav Golyanik](https://people.mpi-inf.mpg.de/~golyanik/), [Eddy Ilg](https://cvmp.cs.uni-saarland.de/people/#eddy-ilg)

This project is the source code used for [Quantum-Hybrid Stereo Matching with Nonlinear Regularization and Spatial Pyramids](https://4dqv.mpi-inf.mpg.de/QHSM/). This work was accepted to [3DV 2024](https://3dvconf.github.io/2024/).

## Initial Set Up

This code is written in Python. We recommend [conda](https://docs.conda.io/en/latest/) for ease of set up, however this is not strictly necessary.

Once in our Python environment of choice, run

```
$ pip install -r requirements.txt
```
to install all necessary Python libraries


## DWave

In order to have access to DWave's remote services, you must have an account with DWave, as well as some additional configuration set up on your local machine. Follow the instructions presented [here](https://docs.ocean.dwavesys.com/en/stable/overview/install.html#set-up-your-environment).

## Gurobi

There are additional steps needed to set up Gurobi for optimizations. Start with the guide [here](https://www.gurobi.com/resources/starting-with-gurobi/) for set up instructions.


## iViz 

Optionally, one can also install [iViz](https://github.com/eddy-ilg/iviz) to view relevant data quickly. Alternatively, however, one can view the outputs via standard image viewing applications without any trouble.

## Data

The Middlebury data which we work with in our paper is all available [here](https://vision.middlebury.edu/stereo/data/scenes2001/). Due to the small size of this data, we have included it in the repository.


If you wish to work with the Sintel stereo data, download the .zip from [here](http://sintel.is.tue.mpg.de/stereo), and place it into the /datasets directory. Next, unzip the file. In Linux, this can be done using the unzip command, for example

```
$ unzip datasets/MPI-Sintel-stereo-training-20150305.zip
```


## To Run:

To see the gurobi optimizer run on the Tsukuba pair from the Middlebury dataset, run

```
$ ./example_tsukuba.sh
```
The output (before filtering) should look like this:

<p align="center">
<img src="readme_media/tsukuba.png" alt="Market2" /></br>
</p>

For the first frame of the Market 2 scene of Sintel, run

```
$ ./example_sintel.sh
```

Here are the results of this estimation visualized with matplotlib:

<p align="center">
<img src="readme_media/market_2.png" alt="Market2" /></br>
</p>

One can see the result of running this estimation process across an entire Sintel scene below:

<p align="center">
<img src="readme_media/alley_2.gif" alt="Alley2" /></br>
</p>

## Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{braunstein2023quantumhybrid,
      title={Quantum-Hybrid Stereo Matching With Nonlinear Regularization and Spatial Pyramids}, 
      booktitle={International Conference on 3D Vision (3DV)}, 
      author={Cameron Braunstein and Eddy Ilg and Vladislav Golyanik},
      year={2024}
}
``` 

## License

This work is under the  [MIT License](https://opensource.org/license/mit/).