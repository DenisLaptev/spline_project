------------------------------------Smoothing util

1. task.txt 		= file with tasks list from Yura concerning smoothing util.
2. requirements.txt 	= file with dependencies used in this util.
3. initial images 	= folder with dataset with 3 types of files (txt-file with x,y coordinates of lanes, png-images with colored lanes,png-images with colored planes).
4. research 		= folder with research on smoothing util (not necessary for the final version).
5. Smoothed 		= folder with txt-files with x,y coordinates of smoothed lanes. This is a result of the smoothing of initial lanes from dataset.
6. task1 		= folder with materials on task1. In task1 smoothing is made only for 239,544,594,614,877,2305,2316,2616 frames.
7. task2_all_images 	=  folder with materials on task2. In task2 smoothing is made for ALL dataset.
8. README	 	= this file.


------------------------------------task1

6-1. get_smoothed_poly_lanes.py 	= script that makes smoothing using polynomials of the 2,3,4,5 order and plots corresponding images.
6-2. get_results_metrix.py 	= script that 
	1) makes smoothing using polynomials of the 3,4 order and plots corresponding images; 
	2) builds binary images of smoothed lanes;
	3) calculates metrix (difference in pixels between two graphs, area in pixels between two graphs) and plots corresponding images;

6-3. images_for_task1 		= folder with images for task1.
6-4. results_smoothed_lanes 	= folder with results of script get_smoothed_poly_lanes.py (1.)
6-5. results_metrix		= folder with results of script get_results_metrix.py (2.)
6-6. results_binary_images	= folder with results of script get_results_metrix.py (2.) (binary images of smoothed lanes)

------------------------------------task2_all_images

7-1. parsing_script.py		= script which takes ALL dataset (x,y coordinates of all lanes) and returns SMOOTHED x,y coordinates of all lanes and saves them in the Smoothed folder(5.)
7-2. show_images_script.py	= script which shows result (all smoothed lanes from "Smoothed" folder (5.))

 
