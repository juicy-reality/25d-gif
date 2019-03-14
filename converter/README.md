## Extract frames

	extract.py bike.mp4 bike --start 106 --count 112 --skip 2

## Join color

	python3 joincolor.py "bike/color/*.jpg" bike_color.jpg

## Join depth

	python3 joindepth.py "bike/depthn/*.npy" bike_depth.png
