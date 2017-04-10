for i in {1..5000}
do 
	 ORIGINAL="/home/jason/Documents/CMPS-4720-6720/Dataset/Original512/Original_$i.jpg"
	 Expert="/home/jason/Documents/CMPS-4720-6720/Dataset/Expert_E/ExpertE_$i.jpg"
	 output="/home/jason/Documents/CMPS-4720-6720/Dataset/ExpE512/OrigB_$i.jpg"
	 convert $ORIGINAL $Expert +append $output


done

