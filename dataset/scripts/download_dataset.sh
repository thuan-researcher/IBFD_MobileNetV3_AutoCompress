#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

dataset_root=${1}

if [ "$#" -ne 1 ]; then
     echo "Usage: ./download_dataset.sh DATASET_DOWNLOAD_PATH"
     exit 2
fi

# Check if directory exists
if [[ ! -d "${dataset_root}" ]]
then	
 	echo "Invalid destination path: ${dataset_root}."	
	exit 2
fi


# echo "" 
# echo "Downloading dataset to ${dataset_root}"
# echo ""
# filename="anime_face_dataset.zip"
# url="https://subunmedu-my.sharepoint.com/:u:/g/personal/phuongdong_unicornoffice_net/ETddijshXgpBmb_KcCmHHQwBGZWc0uAfJpPJ4nSLdzqcuA?download=1"
# wget --no-check-certificate ${url} -O ${dataset_root}/${filename}

echo "" 
echo "Downloading dataset to ${dataset_root}"
echo ""
filename="IBFD_dataset.zip"

# url="https://drive.google.com/uc?export=download&id=1LSIZKC1XvmRA4nVnEzIbe6BllwbQvG6z"
url="https://drive.google.com/uc?export=download&id=1wDmHZyrx76lqZkTGpMVyY3RcSYPBo4-4&confirm=t"
wget --no-check-certificate ${url} -O ${dataset_root}/${filename}

echo ""
echo "Extract dataset (takes > 10 mins)"
echo ""
unzip ${dataset_root}/${filename} -d ${dataset_root}


# symlink_dst=$(dirname ${BASH_SOURCE[0]})
# echo ""
# echo "Add a symbolic link to : ${symlink_dst}/../anime_face"
# echo ""
# ln -s ${dataset_root}/Set5 ../Set5


rm -r -f ${dataset_root}/${filename}

echo ""
echo "Done."
echo "" 