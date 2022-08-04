#! /usr/bin/bash

# origin directory & sub directories
odir=/run/user/120763473/gvfs/smb-share:server=coeciv-nas05.cec.sc.edu,share=goharian/projects/horus/cameras/beena/records;

# destination directory
ddir=/home/serfani/Downloads/horus/machine_learning/dataset;

dirs=("2022-03-31" "2022-04-05");
ntrains=(151 4);
nvals=(22 1);
ntests=(44 2);

for i in "${!dirs[@]}"; do 
	echo "---> $i";
	dir=${dirs[$i]}; ntrain=${ntrains[$i]}; nval=${nvals[$i]}; ntest=${ntests[$i]};
	# echo $dir; echo $ntrain; echo $nval; echo $ntest;
	# find ${odir}/${dir}/images/*.jpg | cut -d "/" -f 14; exit 
	
	find ${odir}/${dir}/images/*.jpg | cut -d "/" -f 14 | shuf > ${odir}/${dir}/images_list.txt;
	cat ${odir}/${dir}/images_list.txt | cut -d "." -f 1 | sed 's/$/.png/' > ${odir}/${dir}/masks_list.txt;
	

	cat ${odir}/${dir}/images_list.txt | head -n ${ntrain} > ${odir}/${dir}/images_train_list.txt; 
	cat ${odir}/${dir}/images_list.txt | tail -n ${ntest} > ${odir}/${dir}/images_test_list.txt; 
	cat ${odir}/${dir}/images_list.txt | head -n -${ntest} | tail -n ${nval} > ${odir}/${dir}/images_val_list.txt;

	cat ${odir}/${dir}/masks_list.txt | head -n ${ntrain} > ${odir}/${dir}/masks_train_list.txt; 
	cat ${odir}/${dir}/masks_list.txt | tail -n ${ntest} > ${odir}/${dir}/masks_test_list.txt;
	cat ${odir}/${dir}/masks_list.txt | head -n -${ntest} | tail -n ${nval} > ${odir}/${dir}/masks_val_list.txt;
	

	rsync --files-from=${odir}/${dir}/images_train_list.txt ${odir}/${dir}/images ${ddir}/images/train;
	rsync --files-from=${odir}/${dir}/images_test_list.txt ${odir}/${dir}/images ${ddir}/images/test;
	rsync --files-from=${odir}/${dir}/images_val_list.txt ${odir}/${dir}/images ${ddir}/images/val;
	
	rsync --files-from=${odir}/${dir}/masks_train_list.txt ${odir}/${dir}/masks/SegmentationClass ${ddir}/masks/train/;
	rsync --files-from=${odir}/${dir}/masks_test_list.txt ${odir}/${dir}/masks/SegmentationClass ${ddir}/masks/test/;
	rsync --files-from=${odir}/${dir}/masks_val_list.txt ${odir}/${dir}/masks/SegmentationClass ${ddir}/masks/val/;
done	
