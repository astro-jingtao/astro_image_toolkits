#PBS -N compare_with_direct
#PBS -j oe
#PBS -l select=1:ncpus=5:mem=92gb
source activate science
cd /home/jingtao/software/workspace/myrepos/astro_image_toolkits/benchmark/conv_fft_nan
python compare_with_direct.py
