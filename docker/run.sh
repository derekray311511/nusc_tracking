xhost +local:

docker run \
-it \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--network host \
--rm \
--name tracking \
-e GRANT_SUDO=yes \
-v /data/Nuscenes/CBMOT/data:/data \
-v /data/Nuscenes/CBMOT_update:/home/Student/Tracking \
tracking \
bash
