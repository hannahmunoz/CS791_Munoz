#!/bin/bash

NUMTRIALS=1
MAXTHREADS=1
THREADSTEP=128
INITALMATRIX=500
BATCHFILE="./seq_matrix_mult.sh"

echo "Dimension of Matrix, Blocks, Threads, Time" > seq_data.csv

for((matrixdim = $INITALMATRIX; matrixdim <= 4000; matrixdim+=500))
do
	#sbatch has odd requirements that the number of nodes be declared
	# upfront when queueing files, so seperate files with different
	# numbers of nodes must be hardcoded in
	#
	#This tells us what node configuration batchfile to use
	# all other arguments can be passed in as cmdline args


		for ((thread = 1; thread <= $MAXTHREADS; thread+=THREADSTEP))
		do

			for ((trial = 0; trial < $NUMTRIALS; trial+=1 ))
			do
					#queries squeue for only usernames and stores result in TEST
					TEST=$(squeue -o"%.20u")

					#checks test against my username and
					# loops while username is found in this test string
					while [[ "$TEST"  =~  "cscully" ]]
					do
						sleep 1s
						TEST=$(squeue -o"%.20u")
					done

					let BLOCKS=$matrixdim


					#sbatch file loaded with some command line arguments
					#if [[ $(($BLOCKS)) == $matrixdim ]]; then
							$BATCHFILE $matrixdim
					#else
					#	  echo "$matrixdim, $BLOCKS, $thread, N/A" >> extra_data.csv
					#fi

					#sleep 1s

			done


			if [ $thread -eq 1 ]; then
				thread=0
			fi
		done
done
