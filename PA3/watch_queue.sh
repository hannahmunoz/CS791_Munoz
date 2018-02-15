
for(( ; ; ))
do
  clear
  squeue -o"%.18i %.9P %.20j %.8u %.8T %.10M %.9l %.6D %R %p"
  echo "ctrl+c to quit"
  sleep 1s
done
