for i in {1..5000}
do 
	 ORIGINAL="/home/jason/Documents/CMPS-4720-6720/Dataset/Orig512_People/Original_$i.jpg"
	 Expert="/home/jason/Documents/CMPS-4720-6720/Dataset/ExpE_People/ExpertE_$i.jpg"
	 output="/home/jason/Documents/CMPS-4720-6720/Dataset/PeopleExpEOrig/OrigB_$i.jpg"
	 convert $ORIGINAL $Expert +append $output


done

