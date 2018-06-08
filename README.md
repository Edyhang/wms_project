# wms_project
1. Our data is downloaded from MIT Physiobank Fantasia and MGH/MF databases (https://www.physionet.org/).

2. The subjects we picked from Fantasia database are listed as follows: 'f2o01', 'f2o03', 'f2o04', 'f2o06', 'f2o07', 'f2y01', 'f2y02', 'f2y03', 'f2y04', 'f2y05', 'f2y06', 'f2y07'.

3. The subjects we picked from MGH/MF database are listed as follows: 'mgh001', 'mgh010', 'mgh016', 'mgh019', 'mgh029', 'mgh033', 'mgh035', 'mgh036', 'mgh051', 'mgh052', 'mgh069', 'mgh079', 'mgh087', 'mgh088', 'mgh098', 'mgh102', 'mgh105', 'mgh129', 'mgh143', 'mgh191', 'mgh195'.

4. As Fantasia and MGH/MF databases have different sampling rate for the physiological signals, we upsampled Fantasia to 360Hz (same sampling rate as the signals in MGH/MF) databases so that we have the same sampling rate for all the subjects.

5. The code contains both detection models with final parameter settings, one is time series-based detection system, another one is image-based detection system.
