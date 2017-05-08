# Probabalistic Tensor Networks
This is the repository for the Matlab reference implementation of Probabalistic Tensor Networks, as described in my thesis. The main code is contained in the Matlab package PTN (folder named "+PTN"); required dependencies are also [included](#included-packages-and-third-party-code). The results of the automated tests conducted in chapter 4 are included under the "results" folder; if you want to try reproducing them and the related figures in my thesis, see the [Usage](#usage) section of the current document.

## Installation
Either clone the repository from GitHub, or use the download as zip option and extract. Add the root folder with subfolders to the Matlab path. No additional Matlab toolboxes should be required, although the PTN learning algorithm benefits substantially if the [Parallel Computing Toolbox]((https://www.mathworks.com/products/parallel-computing.html)) is available. I have only tested the code with Matlab R2013a and R2016a (on Windows 7 and Windows 10), though it will likely be compatible with earlier versions.

## Usage
The main code is included as the Matlab package PTN. For example, a QPTN object representing the network shown in Figure 4.6.1a on page 200 of the thesis can be created with the command

``` matlab
Q = PTN.qptn([2 2], {[2 2], [2 2]}, [1 1 2 1], {[1 2], [2]}, [], 3);
```

The main algorithm is contained in [+PTN/learnPTN.m](./+PTN/learnPTN.m). It expects a QPTN and a data set, which should be a positive tensor of the same shape as the output shape of the QPTN. The additional optional parameters are the number of parallel workers to use (if the [Matlab Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html) is available, this can speed the algorithm up substantially at the cost of marginal additional memory usage) and number of alternating update iterations. See the comments in the [code](./+PTN/learnPTN.m) for a more detailed explanation.

Example usage:
``` matlab
targetDist = PTN.test.randSimplex([2 2]);
data = PTN.draw_from_JPT(targetDist, 100);
[learnedDist, learnedParameters] = PTN.learnPTN(Q, data);
```

Scripts for testing, including scripts for reproducing the results and figures of chapter 4, are included in the subpackage "+test". To recreate the figures from chapter 4, run

``` matlab
PTN.test.chapter_4_tests;
PTN.test.chapter_4_figures;
```

Note that the tests can take quite some time to run; if they are interrupted for any reason, simply re-run the script from the same working directory, and it will automatically resume from the last completed trial. Details of the test conditions and test definition and result file formats are outlined in the comments for the [main testing script](./+PTN/+test/test_ptn_learn.m). 

## Included Packages and Third-Party Code
External packages and code are included for convenience, as well to ensure version compatibility with the thesis code. Version and license information is outlined below, as well as links to the package webpages where applicable.

* [TT-Toolbox](https://github.com/oseledets/TT-Toolbox/)

   The Tensor Train Matlab Toolbox, by Ivan Oseledets, Sergey Dolgov, Vladimir Kazeev, Thomas Mach, Olga Lebedeva, Dmitry Savostyanov, Pavel Zhlobich, and Le Song. Used in this project for the tensor train format and operations, as well as the AMEn algorithim for point-wise function application to tensor trains. The PTN project also uses a [modified version](./+PTN/sparse_exp_normcore) of amen_cross.m that has been specialized to handle point-wise exponentiation of tensor trains that may have extremely negative values. The project is licensed under the [MIT License](https://github.com/oseledets/TT-Toolbox/blob/master/LICENSE). The version included [here](./TT-Toolbox) was forked from the [repository](https://github.com/oseledets/TT-Toolbox/) on August 2, 2015. 

* [ht_tensor](http://anchp.epfl.ch/htucker)

   The Hierarchical Tucker Toolbox, by Christine Tobler. Used in this project for several basic tensor operations. The project is licensed under the [FreeBSD License](./ht_tensor/COPYRIGHT.txt). The version included [here](./ht_tensor) is version 0.8.1 (downloaded April 14, 2011).
   
* [JSONlab](http://iso2mesh.sourceforge.net/cgi-bin/index.cgi?jsonlab)

   JSONlab, by Qianqian Fang. Used in this project to serialize automated test descriptions and results. The included version is version 1.5, and was downladed March 30, 2017. The project is licensed under the [2-clause BSD license](./jsonlab-1.5/LICENSE_BSD.txt).

* [Lightspeed](https://github.com/tminka/lightspeed)

   Lightspeed, by Tom Minka. Used in this project for fast log-space tensor contraction, and generating random parameters for the automated tests. The included version is 2.7 (downloaded September 23, 2016), and is licensed for non-commercial use under [Microsoft Research Shared Source](./lightspeed/license.txt).
   
* [flatten.m](https://www.mathworks.com/matlabcentral/fileexchange/27009-flatten-nested-cell-arrays?focused=6787739&tab=function)

   Small routine to linearize a nested cell array, by Manu Raghavan, redistributed under the following [license](https://www.mathworks.com/matlabcentral/fileexchange/27009-flatten-nested-cell-arrays?focused=6787739&tab=function#license_modal)
   
* [bplot.m](https://github.com/erikrtn/dataviz/blob/master/bplot.m)

   Box plot script by Jonathan C. Lansey and Erik Reinertsen. Forked from Reinertsen's github on April 21, 2017, which I modified slightly for better appearance on the semilog plots used in my thesis. Original file is available from [MathWorks File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/42470-box-and-whiskers-plot--without-statistics-toolbox-), under [this license](https://www.mathworks.com/matlabcentral/fileexchange/42470-box-and-whiskers-plot--without-statistics-toolbox-#license_modal).