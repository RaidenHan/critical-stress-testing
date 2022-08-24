file_list=("update_data.py" "select_features.py" "predict_features.py" "predict_target.py" "data_visualization.py")

for py_file in "${file_list[@]}"
do
	python ${py_file}
done
