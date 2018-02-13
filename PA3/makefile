# Point to includes of local directories
CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
#O_FILES=main.o multiply.o 
SEQ_FILES=sequential.o


#all: $(O_FILES)
#	$(CC) -o PA3 $(O_FILES)

#main.o: ../src/main.cu
#	$(CC) -c ../src/main.cu -o main.o $(INCLUDES)

#multiply.o: ../src/multiply.cu
#	$(CC) -c ../src/multiply.cu -o multiply.o $(INCLUDES)


sequential.o: ../src/sequential.cu
	$(CC) -c ../src/sequential.cu -o sequential.o $(INCLUDES)
	$(CC) -o Seq $(SEQ_FILES)

clean:
	rm -f *.o
	rm -f *~
