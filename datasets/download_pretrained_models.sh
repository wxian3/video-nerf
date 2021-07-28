#!/bin/bash

NAME=$1

if [[ $NAME != "cat_1" && $NAME != "cat_2" && $NAME != "boy_1" &&  $NAME != "boy_2" &&  $NAME != "boy_3" &&  $NAME != "dog" &&  $NAME != "person_1" &&  $NAME != "person_2" &&  $NAME != "person_3" && $NAME != "person_4" && $NAME != "all" ]]; then
    echo "Available videos are: cat_1 cat_2 boy_1 boy_2 boy_3 dog person_1 person_2 person_3 person_4"
    exit 1
fi

if [[ $NAME == "all" ]]; then
  declare -a NAMES=("cat_1" "cat_2" "boy_1" "boy_2" "boy_3" "dog" "person_1" "person_2" "person_3" "person_4")
else
  declare -a NAMES=($NAME)
fi

for NAME in "${NAMES[@]}"
do
  echo "Specified [$NAME]"
  URL=https://www.cs.cornell.edu/~wenqixian/video_nerf/model/$NAME.zip
  ZIP_FILE=./datasets/$NAME.zip
  TARGET_DIR=./datasets/$NAME/
  wget -N $URL -O $ZIP_FILE
  mkdir $TARGET_DIR
  unzip $ZIP_FILE -d ./datasets/
  rm $ZIP_FILE
done
